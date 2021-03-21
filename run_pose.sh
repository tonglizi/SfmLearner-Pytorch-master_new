models=(
data,b16,lr0.0004/03-05-09:54
data,b16,lr0.0004/03-10-09:50
data,b16,lr0.0004/03-09-15:31
data,b16,lr0.0004/03-04-19:34
data,b16,lr0.0004/03-08-11:21
data,b16,lr0.0001/03-06-18:58
data,b16,lr0.0004/03-09-11:53
data,b16,lr0.0004,m0.2/01-18-16:09
data,500epochs,epoch_size3000,b32,m0.2/06-17-04_17
)
datasets=(00 02 05 08)
for i in ${models[@]};do
	for j in ${datasets[@]};do
		python3.6 test_VO_pose.py /home/cx/SLAM/SfmLearner-Pytorch-master_new/checkpoints/$i/exp_pose_model_best.pth.tar --dataset-dir /home/sda/dataset --sequences $j --isKitti True
	done
done
