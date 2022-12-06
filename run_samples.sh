python run_pose_estimation.py -j example_files/example_1person/output_json -v example_files/example_1person/short_video.mp4 -o output/`date +%Y-%m-%d_%H%M%S` -u -n 1 -s 21 2 -c 0.7 -m -V -tn '001' -p "Abby"
python run_pose_estimation.py -j example_files/example_3people/output_json -v example_files/example_3people/short_video.mp4 -o output/`date +%Y-%m-%d` -u -n 3 -s 21 2 -c 0.7 -m -V -tn 'test123' -p "Abby" -p "Brad" -p "Charlie"



python run_pose_estimation.py -j error_analysis/output/2022-12-01_145048/json -o output/`date +%Y-%m-%d_%H%M%S` -b "Nose" -n 5 -s 13 2 -c 0.7 -tn '001'

python run_pose_estimation.py -j example_files/example_3people/output_json -o output/`date +%Y-%m-%d_%H%M%S` -u -n 3 -s 21 2 -c 0.7 -m -V -tn '001'