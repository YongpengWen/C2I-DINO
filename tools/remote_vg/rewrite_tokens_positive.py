import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_REMOTE_VG_CLASSES = (
    # RSVG
    'ground track field',
    'basketball court',
    'baseball field',
    'football field',
    'swimming pool',
    'storage tank',
    'water tower',
    'wastewater plant',
    'parking lot',
    'tennis court',
    'roundabout',
    'building',
    'bridge',
    'river',
    'lake',
    'pond',
    'road',
    'tree',
    # DIOR-RSVG
    'expressway service area',
    'expressway toll station',
    'train station',
    'golf field',
    'airplane',
    'airport',
    'chimney',
    'dam',
    'harbor',
    'overpass',
    'ship',
    'stadium',
    'vehicle',
    'windmill',
)

COMMAND_PREFIX_RE = re.compile(
    r'^\s*(?:find|search|locate|identify)\s+', re.IGNORECASE)

DIOR_CLASS_MAP = {
    'airplane': 'airplane',
    'airport': 'airport',
    'baseballfield': 'baseball field',
    'basketballcourt': 'basketball court',
    'bridge': 'bridge',
    'chimney': 'chimney',
    'dam': 'dam',
    'expressway-service-area': 'expressway service area',
    'expressway-toll-station': 'expressway toll station',
    'golffield': 'golf field',
    'groundtrackfield': 'ground track field',
    'harbor': 'harbor',
    'overpass': 'overpass',
    'ship': 'ship',
    'stadium': 'stadium',
    'storagetank': 'storage tank',
    'tenniscourt': 'tennis court',
    'trainstation': 'train station',
    'vehicle': 'vehicle',
    'windmill': 'windmill',
}

RSVG_FILENAME_CLASS_MAP = {
    'junction_roundabout': 'roundabout',
    'leisure_track': 'ground track field',
    'man_made_bridge': 'bridge',
    'man_made_storage_tank': 'storage tank',
    'sport_baseball': 'baseball field',
    'sport_basketball': 'basketball court',
    'sport_soccer': 'football field',
    'sport_swimming': 'swimming pool',
    'sport_tennis': 'tennis court',
}

SPAN_PREFIX_BOUNDARY_RE = re.compile(
    r'(?:[.,;]\s+|\b(?:has|have|contains?|including)\b\s+)',
    re.IGNORECASE)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Rewrite ODVG and MDETR/RefCOCO-style tokens_positive '
        'from full expressions to target noun phrase spans for remote VG '
        'datasets such as RSVG and DIOR-RSVG.')
    parser.add_argument(
        '--data-root',
        required=True,
        help='Dataset root that contains annotation directories.')
    parser.add_argument(
        '--odvg-ann-dir',
        default=None,
        help='Input ODVG directory under DATA_ROOT. If omitted, ODVG is skipped.')
    parser.add_argument(
        '--mdetr-ann-dir',
        default=None,
        help='Input MDETR/COCO directory under DATA_ROOT. If omitted, MDETR is '
        'skipped.')
    parser.add_argument(
        '--mdetr-prefix',
        default='finetune_rsvg',
        help='MDETR filename prefix, e.g. finetune_rsvg or '
        'finetune_dior_rsvg.')
    parser.add_argument(
        '--out-suffix',
        default='_target_np',
        help='Suffix appended to generated annotation directories.')
    parser.add_argument(
        '--class-file',
        default=None,
        help='Optional text file with one target class per line.')
    parser.add_argument(
        '--classes',
        nargs='*',
        default=None,
        help='Optional extra target class names.')
    parser.add_argument(
        '--dataset',
        choices=('auto', 'rsvg', 'dior_rsvg'),
        default='auto',
        help='Dataset-specific source for target_class.')
    parser.add_argument(
        '--xml-ann-dir',
        default='Annotations',
        help='XML annotation directory under DATA_ROOT for DIOR-RSVG.')
    return parser.parse_args()


def load_class_names(class_file=None, extra_classes=None):
    classes = list(DEFAULT_REMOTE_VG_CLASSES)
    if class_file:
        with open(class_file, 'r', encoding='utf-8') as f:
            classes.extend(line.strip() for line in f if line.strip())
    if extra_classes:
        classes.extend(extra_classes)

    normalized = []
    seen = set()
    for class_name in classes:
        class_name = ' '.join(class_name.lower().split())
        if class_name and class_name not in seen:
            normalized.append(class_name)
            seen.add(class_name)
    return sorted(normalized, key=lambda x: (-len(x), x))


def plural_pattern(class_name):
    words = class_name.split()
    last_word = words[-1]
    escaped_prefix = [re.escape(word) for word in words[:-1]]
    escaped_last = re.escape(last_word)
    if last_word.endswith('y'):
        escaped_last = f'(?:{re.escape(last_word)}|{re.escape(last_word[:-1])}ies)'
    elif last_word.endswith('s'):
        escaped_last = f'{escaped_last}(?:es)?'
    else:
        escaped_last = f'{escaped_last}s?'
    return r'\s+'.join(escaped_prefix + [escaped_last])


def normalize_dior_class(raw_class):
    key = raw_class.strip().lower()
    return DIOR_CLASS_MAP.get(key, key.replace('-', ' '))


def get_rsvg_target_class(filename):
    stem = Path(filename).stem.lower()
    parts = stem.split('_')
    for size in range(min(4, len(parts)), 0, -1):
        key = '_'.join(parts[-size:])
        if key in RSVG_FILENAME_CLASS_MAP:
            return RSVG_FILENAME_CLASS_MAP[key]
    return None


def iter_dior_xml_objects(ann_dir):
    xml_paths = sorted(Path(ann_dir).glob('*.xml'))
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext('./filename')
        for obj in root.findall('object'):
            box = obj.find('bndbox')
            phrase = (obj.findtext('description') or '').strip()
            bbox = [
                float(box.findtext('xmin')),
                float(box.findtext('ymin')),
                float(box.findtext('xmax')),
                float(box.findtext('ymax')),
            ]
            yield {
                'filename': filename,
                'phrase': phrase,
                'bbox': bbox,
                'target_class': normalize_dior_class(obj.findtext('name') or ''),
            }


def build_dior_target_maps(data_root, xml_ann_dir):
    objects = list(iter_dior_xml_objects(Path(data_root) / xml_ann_dir))
    by_index = {idx: obj['target_class'] for idx, obj in enumerate(objects)}
    by_key = {}
    for obj in objects:
        key = (
            obj['filename'],
            obj['phrase'],
            tuple(round(v, 3) for v in obj['bbox']),
        )
        by_key[key] = obj['target_class']
    return by_index, by_key


def infer_dataset(args, mdetr_prefix):
    if args.dataset != 'auto':
        return args.dataset
    if 'dior' in mdetr_prefix.lower() or 'dior' in str(args.data_root).lower():
        return 'dior_rsvg'
    return 'rsvg'


def find_target_span(caption, target_class, class_names):
    lower = caption.lower()
    matches = []
    search_classes = [target_class] if target_class else class_names
    for class_name in search_classes:
        pattern = r'(?<![a-z])' + plural_pattern(class_name) + r'(?![a-z])'
        match = re.search(pattern, lower)
        if match:
            matches.append((match.start(), match.end(), class_name))

    if not matches:
        raise ValueError(f'No known target class found: {caption}')

    cls_start, cls_end, matched_class = min(
        matches, key=lambda x: (x[0], -(x[1] - x[0])))
    prefix = caption[:cls_start]
    command = COMMAND_PREFIX_RE.match(prefix)
    start = command.end() if command else 0
    for boundary in SPAN_PREFIX_BOUNDARY_RE.finditer(prefix):
        start = max(start, boundary.end())

    while start < cls_start and caption[start].isspace():
        start += 1
    while cls_end > start and caption[cls_end - 1].isspace():
        cls_end -= 1
    return start, cls_end, matched_class


def get_odvg_target_class(item, dataset, dior_by_key):
    if dataset == 'rsvg':
        return get_rsvg_target_class(item['filename'])
    if dataset == 'dior_rsvg':
        region = item['grounding']['regions'][0]
        key = (
            item['filename'],
            item['grounding']['caption'],
            tuple(round(float(v), 3) for v in region['bbox']),
        )
        return dior_by_key.get(key)
    return None


def get_mdetr_target_class(image, dataset, dior_by_index):
    if dataset == 'rsvg':
        return get_rsvg_target_class(image['file_name'])
    if dataset == 'dior_rsvg':
        return dior_by_index.get(image.get('source_object_idx'))
    return None


def rewrite_odvg(in_path, out_path, class_names, dataset, dior_by_key):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    total = 0
    with in_path.open('r', encoding='utf-8') as src, out_path.open(
            'w', encoding='utf-8') as dst:
        for line in src:
            item = json.loads(line)
            caption = item['grounding']['caption']
            target_class = get_odvg_target_class(item, dataset, dior_by_key)
            start, end, matched_class = find_target_span(
                caption, target_class, class_names)
            for region in item['grounding']['regions']:
                old = region.get('tokens_positive')
                new = [[start, end]]
                region['tokens_positive'] = new
                region['phrase'] = caption[start:end]
                region['target_class'] = target_class or matched_class
                region['target_phrase'] = caption[start:end]
                region['target_tokens_positive'] = new
                region['caption_suffix'] = target_class or matched_class
                changed += old != new
                total += 1
            item['grounding']['target_class'] = target_class or matched_class
            item['grounding']['caption_suffix'] = target_class or matched_class
            dst.write(json.dumps(item, ensure_ascii=False) + '\n')
    return total, changed


def rewrite_mdetr(in_path, out_path, class_names, dataset, dior_by_index):
    data = json.loads(in_path.read_text(encoding='utf-8'))
    image_by_id = {image['id']: image for image in data['images']}
    changed = 0
    total = 0
    for ann in data['annotations']:
        image = image_by_id[ann['image_id']]
        caption = image['caption']
        target_class = get_mdetr_target_class(image, dataset, dior_by_index)
        start, end, matched_class = find_target_span(
            caption, target_class, class_names)
        old = ann.get('tokens_positive')
        new = [[start, end]]
        ann['tokens_positive'] = new
        ann['target_class'] = target_class or matched_class
        ann['target_phrase'] = caption[start:end]
        ann['target_tokens_positive'] = new
        ann['caption_suffix'] = target_class or matched_class
        image['target_class'] = target_class or matched_class
        image['target_phrase'] = caption[start:end]
        image['caption_suffix'] = target_class or matched_class
        changed += old != new
        total += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    return total, changed


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    class_names = load_class_names(args.class_file, args.classes)
    dataset = infer_dataset(args, args.mdetr_prefix)
    dior_by_index = {}
    dior_by_key = {}
    if dataset == 'dior_rsvg':
        dior_by_index, dior_by_key = build_dior_target_maps(
            data_root, args.xml_ann_dir)
    print(f'dataset={dataset}, classes={len(class_names)}')

    if args.odvg_ann_dir:
        odvg_in = data_root / args.odvg_ann_dir
        odvg_out = data_root / f'{args.odvg_ann_dir}{args.out_suffix}'
        for in_path in sorted(odvg_in.glob('*_vg.jsonl')):
            split = in_path.name[:-len('_vg.jsonl')]
            total, changed = rewrite_odvg(
                in_path,
                odvg_out / f'{split}_vg.jsonl',
                class_names,
                dataset,
                dior_by_key)
            print(f'ODVG {split}: regions={total}, changed={changed}')

    if args.mdetr_ann_dir:
        mdetr_in = data_root / args.mdetr_ann_dir
        mdetr_out = data_root / f'{args.mdetr_ann_dir}{args.out_suffix}'
        for split in ('train', 'val', 'test'):
            total, changed = rewrite_mdetr(
                mdetr_in / f'{args.mdetr_prefix}_{split}.json',
                mdetr_out / f'{args.mdetr_prefix}_{split}.json',
                class_names,
                dataset,
                dior_by_index)
            print(f'MDETR {split}: annotations={total}, changed={changed}')


if __name__ == '__main__':
    main()
