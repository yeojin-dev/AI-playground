import os

IMAGE_ROOT = './cityscapes/leftImg8bit/{}'
LABEL_ROOT = './cityscapes/gtFine/{}'

for mode in ('train', 'val'):

	with open(f'./cityscapes/{mode}Attribute.txt', 'w') as f:
		for images, labels in zip(os.walk(IMAGE_ROOT.format(mode)), os.walk(LABEL_ROOT.format(mode))):
			image_files = sorted(images[2])
			label_files = sorted([label_file for label_file in labels[2] if label_file.endswith('labelTrainIds.png')])

			images_abs_root = images[0]
			labels_abs_root = labels[0]

			for image_file, label_file in zip(image_files, label_files):
				image_filepath = os.path.join(os.path.abspath(images_abs_root), image_file)
				label_filepath = os.path.join(os.path.abspath(labels_abs_root), label_file)

				f.writelines(f'{image_filepath} {label_filepath}\n')
