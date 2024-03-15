find . -name 'random*' -maxdepth 1 |sort -t '\0' -n

/usr/bin/ffmpeg \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency1.0_octaves1_time2024_03_13-22_52_03.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency1.0_octaves2_time2024_03_13-23_27_21.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency1.0_octaves3_time2024_03_13-23_58_48.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency1.0_octaves4_time2024_03_14-00_30_21.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency2.0_octaves1_time2024_03_14-01_01_57.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency2.0_octaves2_time2024_03_14-01_33_31.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency2.0_octaves3_time2024_03_14-02_05_13.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency2.0_octaves4_time2024_03_14-02_36_39.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency4.0_octaves1_time2024_03_14-03_08_05.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency4.0_octaves2_time2024_03_14-03_39_30.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency4.0_octaves3_time2024_03_14-04_10_57.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency4.0_octaves4_time2024_03_14-04_42_20.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency8.0_octaves1_time2024_03_14-05_13_40.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency8.0_octaves2_time2024_03_14-05_45_00.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency8.0_octaves3_time2024_03_14-06_16_18.gif \
-i ./random_periodic_256x256_u2_k1_TGVFalse_Re4000.0_dt0.001_L1_seed12968534_baseFrequency8.0_octaves4_time2024_03_14-06_47_29.gif \
-filter_complex \
"
[ 0:v][ 1:v][ 2:v][ 3:v]hstack=inputs=4[r1];\
[ 4:v][ 5:v][ 6:v][ 7:v]hstack=inputs=4[r2];\
[ 8:v][ 9:v][10:v][ 11:v]hstack=inputs=4[r3];\
[12:v][13:v][14:v][ 15:v]hstack=inputs=4[r4];\
[r1][r2][r3][r4]vstack=inputs=4[v]" \
-map "[v]" \
-r 30 -c:v libx264 -b:v 100M combined.mp4

ffmpeg -i combined.mp4 -vf "fps=30,scale=1280:-1:flags=lanczos" -c:v libx264 -b:v 25M combined_720p.mp4
ffmpeg -loglevel warning -hide_banner -y -i combined_720p.mp4 -vf "fps=30,scale=1280:-1:flags=lanczos,palettegen" palette.png
ffmpeg -loglevel warning -hide_banner -y -i combined_720p.mp4 -i palette.png -filter_complex "fps=30,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" combined.gif
