# ------------------------------------------------------------------------
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import argparse
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader

from datasets.vcoco import build as build_dataset
from models.backbone import build_backbone
import util.misc as utils
import json
from pycocotools.coco import COCO
from tqdm import tqdm
import copy


class VCOCOeval(object):

    def __init__(self, vsrl_annot_file, coco_annot_file,
                 split_file):
        """Input:
        vslr_annot_file: path to the vcoco annotations
        coco_annot_file: path to the coco annotations
        split_file: image ids for split
        """
        self.COCO = COCO(coco_annot_file)
        self.VCOCO = _load_vcoco(vsrl_annot_file)
        self.image_ids = np.loadtxt(open(split_file, 'r'))
        # simple check
        assert np.all(np.equal(np.sort(np.unique(self.VCOCO[0]['image_id'])), self.image_ids))

        self._init_coco()
        self._init_vcoco()

    def _init_vcoco(self):
        actions = [x['action_name'] for x in self.VCOCO]
        roles = [x['role_name'] for x in self.VCOCO]
        self.actions = actions
        self.actions_to_id_map = {v: i for i, v in enumerate(self.actions)}
        self.num_actions = len(self.actions)
        self.roles = roles

    def _init_coco(self):
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}

    def _get_vcocodb(self):
        vcocodb = copy.deepcopy(self.COCO.loadImgs(self.image_ids.tolist()))
        for entry in vcocodb:
            self._prep_vcocodb_entry(entry)
            self._add_gt_annotations(entry)

        return vcocodb

    def _prep_vcocodb_entry(self, entry):
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['gt_actions'] = np.empty((0, self.num_actions), dtype=np.int32)
        entry['gt_role_id'] = np.empty((0, self.num_actions, 2), dtype=np.int32)

    def _add_gt_annotations(self, entry):
        ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
        objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_ann_ids = []
        width = entry['width']
        height = entry['height']
        for i, obj in enumerate(objs):
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
            x1, y1, x2, y2 = clip_xyxy_to_image(
                x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_ann_ids.append(ann_ids[i])
        num_valid_objs = len(valid_objs)
        assert num_valid_objs == len(valid_ann_ids)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_actions = -np.ones((num_valid_objs, self.num_actions), dtype=entry['gt_actions'].dtype)
        gt_role_id = -np.ones((num_valid_objs, self.num_actions, 2), dtype=entry['gt_role_id'].dtype)

        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            is_crowd[ix] = obj['iscrowd']

            gt_actions[ix, :], gt_role_id[ix, :, :] = \
                self._get_vsrl_data(valid_ann_ids[ix],
                                    valid_ann_ids, valid_objs)

        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['gt_actions'] = np.append(entry['gt_actions'], gt_actions, axis=0)
        entry['gt_role_id'] = np.append(entry['gt_role_id'], gt_role_id, axis=0)

    def _get_vsrl_data(self, ann_id, ann_ids, objs):
        """ Get VSRL data for ann_id."""
        action_id = -np.ones((self.num_actions), dtype=np.int32)
        role_id = -np.ones((self.num_actions, 2), dtype=np.int32)
        # check if ann_id in vcoco annotations
        in_vcoco = np.where(self.VCOCO[0]['ann_id'] == ann_id)[0]
        if in_vcoco.size > 0:
            action_id[:] = 0
            role_id[:] = -1
        else:
            return action_id, role_id
        for i, x in enumerate(self.VCOCO):
            assert x['action_name'] == self.actions[i]
            has_label = np.where(np.logical_and(x['ann_id'] == ann_id, x['label'] == 1))[0]
            if has_label.size > 0:
                action_id[i] = 1
                assert has_label.size == 1
                rids = x['role_object_id'][has_label]
                assert rids[0, 0] == ann_id
                for j in range(1, rids.shape[1]):
                    if rids[0, j] == 0:
                        # no role
                        continue
                    aid = np.where(ann_ids == rids[0, j])[0]
                    assert aid.size > 0
                    role_id[i, j - 1] = aid
        return action_id, role_id

    def _collect_detections_for_image(self, dets, image_id):

        agents = np.empty((0, 4 + self.num_actions), dtype=np.float32)  # 4 + 26 = 30
        roles = np.empty((0, 5 * self.num_actions, 2), dtype=np.float32)  # (5 * 26), 2
        for det in dets:  # loop all detection instance
            if str(det['image_id']) == str(image_id):  # might be several
                this_agent = np.zeros((1, 4 + self.num_actions), dtype=np.float32)
                this_role = np.zeros((1, 5 * self.num_actions, 2), dtype=np.float32)
                this_agent[0, :4] = det['person_box']
                for aid in range(self.num_actions):  # loop 26 actions
                    for j, rid in enumerate(self.roles[aid]):
                        if rid == 'agent':
                            # if aid == 10:
                            #  this_agent[0, 4 + aid] = det['talk_' + rid]
                            # if aid == 16:
                            #  this_agent[0, 4 + aid] = det['work_' + rid]
                            # if (aid != 10) and (aid != 16):
                            # print(det[self.actions[aid] + '_' + rid])
                            key = self.actions[aid] + '_' + rid
                            if key in det:
                                this_agent[0, 4 + aid] = det[key]
                            else:
                                this_agent[0, 4 + aid] = np.nan
                        else:
                            key = self.actions[aid] + '_' + rid
                            if key in det:
                                this_role[0, 5 * aid: 5 * aid + 5, j - 1] = det[key]
                            else:
                                this_role[0, 5 * aid: 5 * aid + 5, j - 1] = [np.nan, np.nan, np.nan, np.nan, np.nan]
                agents = np.concatenate((agents, this_agent), axis=0)
                roles = np.concatenate((roles, this_role), axis=0)
        return agents, roles

    def _do_eval(self, detections_file, ovr_thresh=0.5, out_name=None):
        vcocodb = self._get_vcocodb()
        self._do_role_eval(vcocodb, detections_file, out_name, ovr_thresh=ovr_thresh, eval_type='scenario_1')

    def _do_role_eval(self, vcocodb, dets, output_txt, ovr_thresh=0.5, eval_type='scenario_1'):
        tp = [[[] for r in range(2)] for a in range(self.num_actions)]
        fp = [[[] for r in range(2)] for a in range(self.num_actions)]
        sc = [[[] for r in range(2)] for a in range(self.num_actions)]

        npos = np.zeros((self.num_actions), dtype=np.float32)

        for i in tqdm(range(len(vcocodb))):
            image_id = vcocodb[i]['id']
            gt_inds = np.where(vcocodb[i]['gt_classes'] == 1)[0]
            # person boxes
            gt_boxes = vcocodb[i]['boxes'][gt_inds]
            gt_actions = vcocodb[i]['gt_actions'][gt_inds]
            # some peorson instances don't have annotated actions
            # we ignore those instances
            ignore = np.any(gt_actions == -1, axis=1)
            assert np.all(gt_actions[np.where(ignore == True)[0]] == -1)

            for aid in range(self.num_actions):
                npos[aid] += np.sum(gt_actions[:, aid] == 1)

            pred_agents, pred_roles = self._collect_detections_for_image(dets, image_id)

            for aid in range(self.num_actions):
                if len(self.roles[aid]) < 2:
                    # if action has no role, then no role AP computed
                    continue

                for rid in range(len(self.roles[aid]) - 1):

                    # keep track of detected instances for each action for each role
                    covered = np.zeros((gt_boxes.shape[0]), dtype=np.bool)

                    # get gt roles for action and role
                    gt_role_inds = vcocodb[i]['gt_role_id'][gt_inds, aid, rid]
                    gt_roles = -np.ones_like(gt_boxes)
                    for j in range(gt_boxes.shape[0]):
                        if gt_role_inds[j] > -1:
                            gt_roles[j] = vcocodb[i]['boxes'][gt_role_inds[j]]

                    agent_boxes = pred_agents[:, :4]
                    role_boxes = pred_roles[:, 5 * aid: 5 * aid + 4, rid]
                    agent_scores = pred_roles[:, 5 * aid + 4, rid]

                    valid = np.where(np.isnan(agent_scores) == False)[0]
                    agent_scores = agent_scores[valid]
                    agent_boxes = agent_boxes[valid, :]
                    role_boxes = role_boxes[valid, :]

                    idx = agent_scores.argsort()[::-1]

                    for j in idx:
                        pred_box = agent_boxes[j, :]
                        overlaps = get_overlap(gt_boxes, pred_box)

                        # matching happens based on the person
                        jmax = overlaps.argmax()
                        ovmax = overlaps.max()

                        # if matched with an instance with no annotations
                        # continue
                        if ignore[jmax]:
                            continue

                        # overlap between predicted role and gt role
                        if np.all(gt_roles[jmax, :] == -1):  # if no gt role
                            if eval_type == 'scenario_1':
                                if np.all(role_boxes[j, :] == 0.0) or np.all(np.isnan(role_boxes[j, :])):
                                    # if no role is predicted, mark it as correct role overlap
                                    ov_role = 1.0
                                else:
                                    # if a role is predicted, mark it as false
                                    ov_role = 0.0
                            elif eval_type == 'scenario_2':
                                # if no gt role, role prediction is always correct, irrespective of the actual predition
                                ov_role = 1.0
                            else:
                                raise ValueError('Unknown eval type')
                        else:
                            ov_role = get_overlap(gt_roles[jmax, :].reshape((1, 4)), role_boxes[j, :])

                        is_true_action = (gt_actions[jmax, aid] == 1)
                        sc[aid][rid].append(agent_scores[j])
                        if is_true_action and (ovmax >= ovr_thresh) and (ov_role >= ovr_thresh):
                            if covered[jmax]:
                                fp[aid][rid].append(1)
                                tp[aid][rid].append(0)
                            else:
                                fp[aid][rid].append(0)
                                tp[aid][rid].append(1)
                                covered[jmax] = True
                        else:
                            fp[aid][rid].append(1)
                            tp[aid][rid].append(0)

        # compute ap for each action
        role_ap = np.zeros((self.num_actions, 2), dtype=np.float32)
        role_ap[:] = np.nan
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2:
                continue
            for rid in range(len(self.roles[aid]) - 1):
                a_fp = np.array(fp[aid][rid], dtype=np.float32)
                a_tp = np.array(tp[aid][rid], dtype=np.float32)
                a_sc = np.array(sc[aid][rid], dtype=np.float32)
                # sort in descending score order
                idx = a_sc.argsort()[::-1]
                a_fp = a_fp[idx]
                a_tp = a_tp[idx]
                a_sc = a_sc[idx]

                a_fp = np.cumsum(a_fp)
                a_tp = np.cumsum(a_tp)
                rec = a_tp / float(npos[aid])
                # check
                assert (np.amax(rec) <= 1)
                prec = a_tp / np.maximum(a_tp + a_fp, np.finfo(np.float64).eps)
                role_ap[aid, rid] = voc_ap(rec, prec)

        f = open(output_txt, 'a')
        print('---------Reporting Role AP (%)------------------')
        f.write('---------Reporting Role AP (%)------------------\n')
        for aid in range(self.num_actions):
            if len(self.roles[aid]) < 2: continue
            for rid in range(len(self.roles[aid]) - 1):
                info = '{: >23}: AP = {:0.2f} (#pos = {:d})'.format(self.actions[aid] + '-' + self.roles[aid][rid + 1],
                                                                    role_ap[aid, rid] * 100.0, int(npos[aid]))
                print(info)
                f.write(info)
                f.write('\n')
        info = 'Average Role [%s] AP = %.2f' % (eval_type, np.nanmean(role_ap) * 100.00)
        print(info)
        f.write(info)
        f.write('\n')
        print('---------------------------------------------')
        f.write('---------------------------------------------\n')
        info = 'Average Role [%s] AP = %.2f, omitting the action "point"' % (
            eval_type, (np.nanmean(role_ap) * 25 - role_ap[-3][0]) / 24 * 100.00)
        print(info)
        f.write(info)
        f.write('\n')
        print('---------------------------------------------')
        f.write('---------------------------------------------\n')
        f.close()


def _load_vcoco(vcoco_file):
    print('loading vcoco annotations...')
    with open(vcoco_file, 'r') as f:
        vsrl_data = json.load(f)
    for i in range(len(vsrl_data)):
        vsrl_data[i]['role_object_id'] = \
            np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']), -1)).T
        for j in ['ann_id', 'label', 'image_id']:
            vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1, 1))
    return vsrl_data


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


def get_overlap(boxes, ref_box):
    ixmin = np.maximum(boxes[:, 0], ref_box[0])
    iymin = np.maximum(boxes[:, 1], ref_box[1])
    ixmax = np.minimum(boxes[:, 2], ref_box[2])
    iymax = np.minimum(boxes[:, 3], ref_box[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((ref_box[2] - ref_box[0] + 1.) * (ref_box[3] - ref_box[1] + 1.) +
           (boxes[:, 2] - boxes[:, 0] + 1.) *
           (boxes[:, 3] - boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec)
    Compute VOC AP given precision and recall.
    [as defined in PASCAL VOC]
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--batch_size', default=2, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_verb_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * HOI
    parser.add_argument('--subject_category_id', default=0, type=int)
    parser.add_argument('--missing_category_id', default=80, type=int)

    parser.add_argument('--hoi_path', type=str)
    parser.add_argument('--param_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--model_name', default='baseline')
    parser.add_argument('--dataset_file', default='vcoco')

    return parser


def main(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)

    valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                     37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                     82, 84, 85, 86, 87, 88, 89, 90)

    verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                    'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                    'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                    'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                    'point_instr', 'read_obj', 'snowboard_instr']

    device = torch.device(args.device)

    dataset_val = build_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    args.lr_backbone = 0
    args.masks = False
    backbone = build_backbone(args)
    if args.model_name == "baseline":
        from models.transformer import build_transformer
        transformer = build_transformer(args)
        from models.hoi import DETRHOI, PostProcessVCOCO
        model = DETRHOI(backbone, transformer, len(valid_obj_ids) + 1, len(verb_classes),
                        args.num_queries, args.num_verb_queries)
    elif args.model_name == "hoi_ts_qpos_eobj_kl":
        from models.transformer import build_hoi_transformer_ts_qpos_eobj_attention_map
        transformer = build_hoi_transformer_ts_qpos_eobj_attention_map(args, begin_l=0,
                                                                       num_obj_classes=args.num_obj_classes,
                                                                       num_verb_classes=args.num_verb_classes)
        from models.ts.hoi_share_qpos_eobj_cos_kl import DETRHOI, PostProcessVCOCO
        model = DETRHOI(
            backbone,
            transformer,
            num_obj_classes=args.num_obj_classes,
            num_verb_classes=args.num_verb_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss
        )
    else:
        print("add model name")
        assert False
    post_processor = PostProcessVCOCO(args.num_queries, args.subject_category_id, dataset_val.correct_mat)
    model.to(device)
    post_processor.to(device)

    checkpoint = torch.load(args.param_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    detections = generate(model, post_processor, data_loader_val, device, verb_classes, args.missing_category_id)
    import os
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    with open(os.path.join(args.save_path, "vcoco.pickle"), 'wb') as f:
        pickle.dump(detections, f, protocol=2)

    vcocoeval = VCOCOeval("data/v-coco/annotations/vcoco_test.json",
                          "data/v-coco/annotations/instances_vcoco_all_2014.json",
                          "data/v-coco/annotations/vcoco_test.ids")
    vcocoeval._do_eval(detections, ovr_thresh=0.5,
                       out_name=os.path.join(args.save_path, "result.txt"))


@torch.no_grad()
def generate(model, post_processor, data_loader, device, verb_classes, missing_category_id):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    detections = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = post_processor(outputs, orig_target_sizes)

        for img_results, img_targets in zip(results, targets):
            for hoi in img_results['hoi_prediction']:
                detection = {
                    'image_id': img_targets['img_id'],
                    'person_box': img_results['predictions'][hoi['subject_id']]['bbox'].tolist()
                }
                if img_results['predictions'][hoi['object_id']]['category_id'] == missing_category_id:
                    object_box = [np.nan, np.nan, np.nan, np.nan]
                else:
                    object_box = img_results['predictions'][hoi['object_id']]['bbox'].tolist()
                cut_agent = 0
                hit_agent = 0
                eat_agent = 0
                for idx, score in zip(hoi['category_id'], hoi['score']):
                    verb_class = verb_classes[idx]
                    score = score.item()
                    if len(verb_class.split('_')) == 1:
                        detection['{}_agent'.format(verb_class)] = score
                    elif 'cut_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        cut_agent = score if score > cut_agent else cut_agent
                    elif 'hit_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        hit_agent = score if score > hit_agent else hit_agent
                    elif 'eat_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        eat_agent = score if score > eat_agent else eat_agent
                    else:
                        detection[verb_class] = object_box + [score]
                        detection['{}_agent'.format(
                            verb_class.replace('_obj', '').replace('_instr', ''))] = score
                detection['cut_agent'] = cut_agent
                detection['hit_agent'] = hit_agent
                detection['eat_agent'] = eat_agent
                detections.append(detection)

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
