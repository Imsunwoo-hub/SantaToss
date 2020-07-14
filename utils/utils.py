import matplotlib.pyplot as plt
import random


# Req. 2-2	세팅 값 저장
def save_config(filename, config):
	with open("./args/" + filename + '_args.txt', 'w') as f: # with 블록을 통해 close()를 호출하지 않고도 파일을 닫을 수 있음
		for arg in vars(config):
			f.write("--%s\t%s\n" % (arg, getattr(config, arg)))

	print("config 파일 저장 완료!")


# Req. 4-1	이미지와 캡션 시각화
def visualize_img_caption(img_root, img_paths, captions, tokenizer):
	n_rand = random.randint(0, len(captions))

	img_path = img_paths[n_rand]
	text = ""
	for i in range(0,len(captions[n_rand])):
		text += "\n"+str(captions[n_rand][i])
		text += "\n"+str(tokenizer[n_rand][i])
	image = plt.imread(img_root + img_path)
	text += "\nimage : " + img_path	#  타이틀에 이미지의 파일명을 출력하기 위해

	plt.figure(figsize=(8, 16))
	plt.title(text)
	plt.imshow(image)
	plt.show()


# Visualize dataset by Daseul
def visualize_dataset(augmented_dataset):
	plt.axis("off")
	for i, augmented in enumerate(augmented_dataset):
		plt.imshow(augmented[0].numpy())
		plt.title("augmented image")
		plt.show()

		if i==2: break