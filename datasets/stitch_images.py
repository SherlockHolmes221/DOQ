import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image as Image
import pickle

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_union_bbox(proc_anno, img, yn):

    # union_bbox = proc_anno['hoi_union_bbox']
    subs = [proc_anno['bbox'][j] for j in proc_anno['hoi_sub']]
    objs = [proc_anno['bbox'][j] for j in proc_anno['hoi_obj']]

    # pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
    # pts2 = np.float32([[0, 0], [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]])
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # image_output = cv2.warpPerspective(image, matrix, (xmax - xmin, ymax - ymin))

    colors = COLORS * 100

    plt.figure()
    plt.imshow(img)

    ax = plt.gca()

    for (sub_xmin, sub_ymin, sub_xmax, sub_ymax), (obj_xmin, obj_ymin, obj_xmax, obj_ymax), c in zip(subs, objs, colors):

        ax.add_patch(plt.Rectangle((sub_xmin, sub_ymin), sub_xmax - sub_xmin, sub_ymax - sub_ymin,
                                   fill=False, color=c, linewidth=1))

        ax.add_patch(plt.Rectangle((obj_xmin, obj_ymin), obj_xmax - obj_xmin, obj_ymax - obj_ymin,
                                   fill=False, color=c, linewidth=1))

    if yn == 1:
        ax.add_patch(plt.Rectangle((union_bbox[0], union_bbox[1]), union_bbox[2] - union_bbox[0], union_bbox[3] - union_bbox[1],
                                   fill=False, color='r', linewidth=3))
    plt.axis('off')

    plt.show()


def get_sim_index(sample_num, nohoi_index, sim_index):
    for i in range(len(nohoi_index)):
        if nohoi_index[i] in sim_index:
            sim_index.remove(nohoi_index[i])
    random_index = np.random.choice(sim_index, sample_num, replace=True)
    return list(random_index)


def get_basic_union_bbox(random_index, annotations, dataset):

    anno_bbox = []  # all gt box in each line
    anno_id = []    # all gt category in each line

    hoi_sub = []  # people index in each line
    hoi_obj = []  # obj index in each line
    verb_id = []  # verb
    hoi_id = []   # verb+obj

    img_name = []

    hoi_union_bbox = []  # union box in each line

    for i in range(len(random_index)):
        annotation = annotations[random_index[i]]['annotations']
        bbox = []
        category_id = []
        for anno in annotation:
            bbox.append(anno['bbox'])
            category_id.append(anno['category_id'])
        anno_bbox.append(bbox)
        anno_id.append(category_id)

        hoi_annotation = annotations[random_index[i]]['hoi_annotation']
        subject_id, object_id, category_id, hoi_category_id = [], [], [], []
        for hoi_anno in hoi_annotation:
            if hoi_anno['object_id'] != -1:
                subject_id.append(hoi_anno['subject_id'])
                object_id.append(hoi_anno['object_id'])
                category_id.append(hoi_anno['category_id'])
                if dataset=='hico':
                    hoi_category_id.append(hoi_anno['hoi_category_id'])
        hoi_sub.append(subject_id)
        hoi_obj.append(object_id)
        verb_id.append(category_id)
        hoi_id.append(hoi_category_id)

        union_bbox = []
        for sub_id, obj_id in zip(hoi_sub[i], hoi_obj[i]):
            try:
                xmin = min(anno_bbox[i][sub_id][0], anno_bbox[i][obj_id][0])
                ymin = min(anno_bbox[i][sub_id][1], anno_bbox[i][obj_id][1])
                xmax = max(anno_bbox[i][sub_id][2], anno_bbox[i][obj_id][2])
                ymax = max(anno_bbox[i][sub_id][3], anno_bbox[i][obj_id][3])
            except:
                raise Exception('current error index is：{}, all index are：{},'
                                'current bbox length is:{},sub_id is:{},obj_id is:{}'.format((i, random_index[i]), random_index, len(anno_bbox[i]), sub_id, obj_id))

            union_bbox.append([xmin, ymin, xmax, ymax])

        hoi_union_bbox.append(union_bbox)

        img_name.append(annotations[random_index[i]]['file_name'])

    anno_dict = {}
    anno_dict['file_name'] = img_name
    anno_dict['bbox'] = anno_bbox
    anno_dict['bbox_id'] = anno_id
    anno_dict['hoi_sub'] = hoi_sub
    anno_dict['hoi_obj'] = hoi_obj
    anno_dict['verb_id'] = verb_id
    anno_dict['hoi_id'] = hoi_id
    anno_dict['hoi_union_bbox'] = hoi_union_bbox

    return anno_dict


def inter_rec(locxx, locyy):
    loc1 = locxx
    loc2 = locyy

    for i in range(0, len(locxx)):
        for j in range(0, len(locxx)):
            if i != j:
                Xmax = max(loc1[i][0], locxx[j][0])
                Ymax = max(loc1[i][1], locxx[j][1])
                M = (Xmax, Ymax)
                Xmin = min(loc2[i][0], locyy[j][0])
                Ymin = min(loc2[i][1], locyy[j][1])
                N = (Xmin, Ymin)
                if M[0] < N[0] and M[1] < N[1]: # determine whether to intersect
                    loc1x = (min(loc1[i][0], locxx[j][0]), min(loc1[i][1], locxx[j][1]))
                    locly = (max(loc2[i][0], locyy[j][0]), max(loc2[i][1], locyy[j][1]))
                    aa = [loc1[i], loc1[j]]
                    bb = [loc2[i], loc2[j]]
                    loc1 = [loc1x if q in aa else q for q in loc1]
                    loc2 = [locly if w in bb else w for w in loc2]

    return loc1, loc2


def combined_union_bbox(locxx, locyy, margin=10):

    locxx = [(i[0], i[1]) for i in locxx]
    locyy = [(i[0], i[1]) for i in locyy]

    finx, finy = inter_rec(locxx, locyy)

    combin = []
    for k in range(len(finx)):
        for v in range(len(finy)):
            if k == v:
                combin.append(finx[k] + finy[v])
    combin = list(set(combin))
    # print('*****************************************************************************')
    # print(combin)
    # print('*****************************************************************************')

    locxx2 = [(i[0] - margin, i[1] - margin) for i in combin]
    locyy2 = [(i[2] + margin, i[3] + margin) for i in combin]

    finx2, finy2 = inter_rec(locxx2, locyy2)

    combin2 = []
    for k in range(len(finx2)):
        for v in range(len(finy2)):
            if k == v:
                combin2.append(finx2[k] + finy2[v])
    combin2 = list(set(combin2))
    return combin2


def get_union_bbox(random_index, annotations, dataset):

    anno_dict = get_basic_union_bbox(random_index, annotations, dataset)

    anno_file_name = anno_dict['file_name']
    anno_bbox = anno_dict['bbox']
    anno_id = anno_dict['bbox_id']
    hoi_sub = anno_dict['hoi_sub']
    hoi_obj = anno_dict['hoi_obj']
    verb_id = anno_dict['verb_id']
    hoi_id = anno_dict['hoi_id']
    hoi_union_bbox = anno_dict['hoi_union_bbox']

    proc_hoi_union_bbox = []
    proc_hoi_sub = []
    proc_hoi_obj = []
    proc_verb_id = []
    proc_hoi_id = []

    for i in range(len(random_index)):  # combine
        xymin = []
        xymax = []
        for j in range(len(hoi_union_bbox[i])):

            xymin.append(tuple(hoi_union_bbox[i][j][0:2]))
            xymax.append(tuple(hoi_union_bbox[i][j][2:4]))

        combin2 = combined_union_bbox(xymin, xymax)  # get combined image
        # print('***************************************************')
        # print(combin2)
        # print('***************************************************')
        proc_hoi_union_bbox.append(combin2)

        hoi_sub_append = []
        hoi_obj_append = []
        verb_id_append = []
        hoi_id_append = []

        for p in range(len(proc_hoi_union_bbox[i])):
            sub, obj, id1, id2 = [], [], [], []
            for q in range(len(hoi_union_bbox[i])):
                if hoi_union_bbox[i][q][0] >= proc_hoi_union_bbox[i][p][0] and hoi_union_bbox[i][q][1] >= proc_hoi_union_bbox[i][p][1] \
                and hoi_union_bbox[i][q][2] <= proc_hoi_union_bbox[i][p][2] and hoi_union_bbox[i][q][3] <= proc_hoi_union_bbox[i][p][3]:
                    sub.append(hoi_sub[i][q])
                    obj.append(hoi_obj[i][q])
                    id1.append(verb_id[i][q])
                    if dataset=='hico':
                        id2.append(hoi_id[i][q])
            hoi_sub_append.append(sub)
            hoi_obj_append.append(obj)
            verb_id_append.append(id1)
            hoi_id_append.append(id2)

        proc_hoi_sub.append(hoi_sub_append)
        proc_hoi_obj.append(hoi_obj_append)
        proc_verb_id.append(verb_id_append)
        proc_hoi_id.append(hoi_id_append)

    proc_anno = []
    for i in range(len(random_index)):
        anno = {}
        t = np.random.randint(len(proc_hoi_union_bbox[i]))
        anno['src_file_name'] = anno_file_name[i]
        anno['bbox'] = anno_bbox[i]
        anno['bbox_id'] = anno_id[i]
        anno['hoi_sub'] = proc_hoi_sub[i][t]
        anno['hoi_obj'] = proc_hoi_obj[i][t]
        anno['hoi_union_bbox'] = [int(v) for v in proc_hoi_union_bbox[i][t]]
        anno['verb_id'] = proc_verb_id[i][t]
        anno['hoi_id'] = proc_hoi_id[i][t]
        proc_anno.append(anno)

    return proc_anno


def get_new_anno_and_image(sample_num, proc_anno, images_folder, margin=10):

    images = []
    for i in range(sample_num):

        proc_anno[i]['hoi_union_bbox'] = [v + margin for v in proc_anno[i]['hoi_union_bbox']]

        for j in range(len(proc_anno[i]['bbox'])):
            proc_anno[i]['bbox'][j] = [v + margin for v in proc_anno[i]['bbox'][j]]

        src_img = Image.open(os.path.join(images_folder, proc_anno[i]['src_file_name']))
        src_img = np.array(src_img)
        pad_img = cv2.copyMakeBorder(src_img, margin, margin, margin, margin, cv2.BORDER_CONSTANT,
                                     value=[128, 128, 128])

        xmin, ymin, xmax, ymax = proc_anno[i]['hoi_union_bbox']
        pts1 = np.float32([[xmin, ymin], [xmax, ymin], [xmin, ymax], [xmax, ymax]])
        pts2 = np.float32([[0, 0], [xmax - xmin, 0], [0, ymax - ymin], [xmax - xmin, ymax - ymin]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        image_output = cv2.warpPerspective(pad_img, matrix, (xmax - xmin, ymax - ymin))
        images.append(image_output)

        union_xmin, union_ymin = proc_anno[i]['hoi_union_bbox'][0], proc_anno[i]['hoi_union_bbox'][1]

        proc_anno[i]['hoi_union_bbox'][0] = proc_anno[i]['hoi_union_bbox'][0] - union_xmin
        proc_anno[i]['hoi_union_bbox'][1] = proc_anno[i]['hoi_union_bbox'][1] - union_ymin
        proc_anno[i]['hoi_union_bbox'][2] = proc_anno[i]['hoi_union_bbox'][2] - union_xmin
        proc_anno[i]['hoi_union_bbox'][3] = proc_anno[i]['hoi_union_bbox'][3] - union_ymin

        for j in range(len(proc_anno[i]['bbox'])):
            proc_anno[i]['bbox'][j][0] = proc_anno[i]['bbox'][j][0] - union_xmin
            proc_anno[i]['bbox'][j][1] = proc_anno[i]['bbox'][j][1] - union_ymin
            proc_anno[i]['bbox'][j][2] = proc_anno[i]['bbox'][j][2] - union_xmin
            proc_anno[i]['bbox'][j][3] = proc_anno[i]['bbox'][j][3] - union_ymin

    return proc_anno, images


def random_flip_horizontal(sample_num, anno, img, p=0.5):

    for i in range(sample_num):
        width = img[i].shape[1]
        if np.random.random() < p:
            if len(img[i].shape) == 3:
                img[i] = img[i][:, ::-1, :]
            elif len(img[i].shape) == 2:
                img[i] = img[i][:, ::-1]
            anno[i]['bbox'] = [[(width-v[2]), v[1], (width-v[0]), v[3]] for v in anno[i]['bbox']]

    return anno, img


def random_rescale(sample_num, anno, img):
    for i in range(sample_num):
        rescale_ratio = np.random.uniform(0.8, 1.25)
        img[i] = cv2.resize(img[i], None, fx=rescale_ratio, fy=rescale_ratio)
        anno[i]['bbox'] = [[v[0]*rescale_ratio, v[1]*rescale_ratio, v[2]*rescale_ratio,
                             v[3]*rescale_ratio] for v in anno[i]['bbox']]

        anno[i]['hoi_union_bbox'] = [v*rescale_ratio for v in anno[i]['hoi_union_bbox']]

    return anno, img


def get_stitch_images(rescale_annos, rescale_images, images_folder):
    to_image = Image.new('RGB', (3000, 3000), color='gray')

    img = [Image.fromarray(v) for v in rescale_images]

    pos = [0] * 4
    max_w_index = sorted(range(4), key=lambda x: img[x].size[0], reverse=True)
    to_image.paste(img[max_w_index[0]], (0, 0))
    pos[max_w_index[0]] = (0, 0)

    to_image.paste(img[max_w_index[3]], (img[max_w_index[0]].size[0], 0))
    pos[max_w_index[3]] = (img[max_w_index[0]].size[0], 0)
    rescale_annos[max_w_index[3]]['bbox'] = [[v[0] + img[max_w_index[0]].size[0], v[1],
                                              v[2] + img[max_w_index[0]].size[0], v[3]] for v in
                                             rescale_annos[max_w_index[3]]['bbox']]

    row1_h = max(img[max_w_index[0]].size[1], img[max_w_index[3]].size[1])

    to_image.paste(img[max_w_index[1]], (0, row1_h))
    pos[max_w_index[1]] = (0, row1_h)
    rescale_annos[max_w_index[1]]['bbox'] = [[v[0], v[1] + row1_h,
                                              v[2], v[3] + row1_h] for v in
                                              rescale_annos[max_w_index[1]]['bbox']]

    to_image.paste(img[max_w_index[2]], (img[max_w_index[1]].size[0], row1_h))
    pos[max_w_index[2]] = (img[max_w_index[1]].size[0], row1_h)
    rescale_annos[max_w_index[2]]['bbox'] = [[v[0] + img[max_w_index[1]].size[0], v[1] + row1_h,
                                              v[2] + img[max_w_index[1]].size[0], v[3] + row1_h] for v in
                                              rescale_annos[max_w_index[2]]['bbox']]

    pos_min_x, pos_min_y, pos_max_x, pos_max_y = [], [], [], []
    for i in range(len(img)):
        pos_min_x.append(pos[i][0])
        pos_min_y.append(pos[i][1])
        pos_max_x.append(pos[i][0] + img[i].size[0])
        pos_max_y.append(pos[i][1] + img[i].size[1])

    xmin, ymin, xmax, ymax = min(pos_min_x), min(pos_min_y), max(pos_max_x), max(pos_max_y)

    image_output = to_image.crop((xmin, ymin, xmax, ymax))

    ratio = image_output.size[0] / image_output.size[1]
    src = []
    src_ratio = []
    for i in range(len(img)):
        src.append(Image.open(os.path.join(images_folder, rescale_annos[i]['src_file_name'])))
        src_ratio.append(src[i].size[0] / src[i].size[1])
    eps = []
    for i in range(len(src_ratio)):
        eps.append(abs(ratio - src_ratio[i]))

    eps_index = sorted(range(len(eps)), key=lambda x: eps[x])[0]

    if eps[eps_index] < 0.5:
        w_ratio = src[eps_index].size[0] / image_output.size[0]
        h_ratio = src[eps_index].size[1] / image_output.size[1]
        image_output = image_output.resize((src[eps_index].size[0], src[eps_index].size[1]))
        for i in range(len(img)):
            rescale_annos[i]['bbox'] = [[v[0] * w_ratio, v[1] * h_ratio, v[2] * w_ratio, v[3] * h_ratio] for v in
                                        rescale_annos[i]['bbox']]
    else:
        w_ratio = src[eps_index].size[0] / image_output.size[0]
        new_h = round(w_ratio * image_output.size[1])
        image_output = image_output.resize((src[eps_index].size[0], new_h))
        for i in range(len(img)):
            rescale_annos[i]['bbox'] = [[v[0] * w_ratio, v[1] * w_ratio, v[2] * w_ratio, v[3] * w_ratio] for v in
                                        rescale_annos[i]['bbox']]

    for i in range(len(img)):
        if i==0:
            pass
        else:
            l = len(rescale_annos[i-1]['bbox'])
            rescale_annos[i]['hoi_sub'] = [v + l for v in rescale_annos[i]['hoi_sub']]
            rescale_annos[i]['hoi_obj'] = [v + l for v in rescale_annos[i]['hoi_obj']]
            rescale_annos[i]['bbox'] = rescale_annos[i-1]['bbox'] + rescale_annos[i]['bbox']
            rescale_annos[i]['bbox_id'] = rescale_annos[i - 1]['bbox_id'] + rescale_annos[i]['bbox_id']
            rescale_annos[i]['hoi_sub'] = rescale_annos[i - 1]['hoi_sub'] + rescale_annos[i]['hoi_sub']
            rescale_annos[i]['hoi_obj'] = rescale_annos[i - 1]['hoi_obj'] + rescale_annos[i]['hoi_obj']
            rescale_annos[i]['verb_id'] = rescale_annos[i - 1]['verb_id'] + rescale_annos[i]['verb_id']
            rescale_annos[i]['hoi_id'] = rescale_annos[i - 1]['hoi_id'] + rescale_annos[i]['hoi_id']
            rescale_annos[i]['src_file_name'] = rescale_annos[i - 1]['src_file_name'] + rescale_annos[i]['src_file_name']

    return rescale_annos[len(rescale_images)-1], image_output


def convanno2hico(anno):
    new_annotations = {}
    new_annotations_annotations = []

    for i in range(len(anno['bbox'])):
        d = {}
        d['bbox'] = anno['bbox'][i]
        d['category_id'] = anno['bbox_id'][i]
        new_annotations_annotations.append(d)
    new_annotations['annotations'] = new_annotations_annotations

    new_annotations_hoi_annotation = []
    for i in range(len(anno['hoi_sub'])):
        d = {}
        d['subject_id'] = anno['hoi_sub'][i]
        d['object_id'] = anno['hoi_obj'][i]
        d['category_id'] = anno['verb_id'][i]
        d['hoi_category_id'] = anno['hoi_id'][i]
        new_annotations_hoi_annotation.append(d)
    new_annotations['hoi_annotation'] = new_annotations_hoi_annotation
    return new_annotations


def convanno2vcoco(anno):

    new_annotations = {}
    new_annotations_annotations = []

    for i in range(len(anno['bbox'])):
        d = {}
        d['bbox'] = anno['bbox'][i]
        d['category_id'] = anno['bbox_id'][i]
        new_annotations_annotations.append(d)
    new_annotations['annotations'] = new_annotations_annotations

    new_annotations_hoi_annotation = []
    for i in range(len(anno['hoi_sub'])):
        d = {}
        d['subject_id'] = anno['hoi_sub'][i]
        d['object_id'] = anno['hoi_obj'][i]
        d['category_id'] = anno['verb_id'][i]
        new_annotations_hoi_annotation.append(d)
    new_annotations['hoi_annotation'] = new_annotations_hoi_annotation
    return new_annotations


def get_replace_image(random_index, annotations, images_folder, dataset_file):

    proc_anno = get_union_bbox(random_index, annotations, dataset_file)
    new_annos, new_imgaes = get_new_anno_and_image(len(random_index), proc_anno, images_folder)
    flip_annos, flip_images = random_flip_horizontal(len(random_index), new_annos, new_imgaes)
    rescale_annos, rescale_images = random_rescale(len(random_index), flip_annos, flip_images)
    anno, image = get_stitch_images(rescale_annos, rescale_images, images_folder)

    if dataset_file == 'vcoco':
        anno = convanno2vcoco(anno)
    elif dataset_file == 'hico':
        anno = convanno2hico(anno)
    else:
        raise Exception('not support dateset')
    return anno, image



if __name__ == '__main__':
    annotations_path = os.path.join('../data/hico_20160224_det', 'annotations', 'trainval_hico.json')
    images_folder = os.path.join('../data/hico_20160224_det', 'images', 'train2015')

    annotations = json.load(open(annotations_path))

    sim_index = pickle.load(open('../datasets/sim_index_hico.pickle', 'rb'))
    nohoi_index = []
    for idx, img_anno in enumerate(annotations):
        for hoi in img_anno['hoi_annotation']:
            if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(
                    img_anno['annotations']) or hoi['subject_id'] >= 100 or hoi['object_id'] >= 100:
                nohoi_index.append(idx)
                break

    idx = np.random.randint(0, len(annotations))
    random_index = [idx] + get_sim_index(3, nohoi_index=nohoi_index, sim_index=sim_index[idx])
    anno, img = get_replace_image(random_index, annotations, images_folder, dataset_file='hico')


