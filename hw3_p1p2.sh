# TODO: create shell script for running your GAN/ACGAN model
echo "-----------start generating picture-------------"
python3 wganDraw.py $1 
python3 acganDraw.py $1
echo "----------finish generating picture-------------"
