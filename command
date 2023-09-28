# train.py
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_1pitch/join_particles.star --simulation -n 100 --cylinder_mask 256 --center_mask 32 --image_patch_size 32
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_1p_fix/join_particles.star -n 10
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_short_test/join_particles.star -n 10
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven/join_particles.star -n 20 --simulation --cylinder_mask 192 -b 2
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation --cylinder_mask 192 -b 2
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation --datadir /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2 --cylinder_mask 192 --center_mask 192 -b 2 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/noise/ctf.pkl
python train.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi/join_particles.star -n 20 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi --cylinder_mask 192 --center_mask 192 -b 2 --ctf_path /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi/noise/ctf.pkl
python train.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi/join_particles.star -n 20 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi --cylinder_mask 192 --center_mask 192 -b 2
python train.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi/join_particles.star -n 20 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi --cylinder_mask 192 --center_mask 192 -b 2
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 100
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 20 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job508/ctf.pkl
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 20 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job508/ctf.pkl --cylinder_mask 128 --loss cross_entropy
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job604/join_particles.star -n 20 --image_patch_size 16
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/join_particles.star --cylinder_mask 96 -n 200 -b 1 --ctf_path /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/ctf.pkl
python train.py /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job245/join_particles.star -n 100
python train.py /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job285/join_particles.star -n 100
python train.py /net/jiang/scratch/li3221/cryodrgn2/EcACC/Subset_J389/particles.star --datadir /net/jiang/scratch/li3221/cryodrgn2/EcACC/Subset_J389 -n 20 --cylinder_mask 128 -b 2 (Apix=5.1375)
python train.py /home/jiang/li3221/scratch/filament_cluster/10340/JoinStar/job505/join_particles.star -n 100 -b 16
python train.py /home/jiang/li3221/scratch/filament_cluster/10340/JoinStar/job368/join_particles.star -b 4 -n 20
python train.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job166/run_it025_data.star -b 16 -n 100
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/join_particles.star --cylinder_mask 128 --image_patch_size 16

python train.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5--cylinder_mask 192 --center_mask 192 -b 2 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5/noise/ctf.pkl
python train_TT.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5 --cylinder_mask 192 --center_mask 192 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5/noise/ctf.pkl
python train_2D.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5 --cylinder_mask 192 --center_mask 192 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi5/noise/ctf.pkl
python train.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2 --cylinder_mask 192 --center_mask 192 -b 2 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2/noise/ctf.pkl
python train_TT.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2 --cylinder_mask 192 --center_mask 192 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2/noise/ctf.pkl
python train_2D.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2 --cylinder_mask 192 --center_mask 192 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi2/noise/ctf.pkl


# train_2D.py
python train_2D.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/join_particles.star --cylinder_mask 128 -n 20 -b 4 --ctf_path /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/ctf.pkl
python train_2D.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation
python train_2D.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 20 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job508/ctf.pkl --cylinder_mask 128


# train_vector.py
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/Class2D/job087/run_it025_data.star -b 16 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job257/run_ct8_it025_data.star -b 32 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job185/run_ct14_it025_data.star -b 32 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job166/run_it025_data.star -b 16 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job387/run_it025_data.star -b 16 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job379/run_it025_data.star -b 16 -n 100
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job537/join_particles.star -b 32 -n 100 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/10230-tau/job537/het_128_mask/z.pkl
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job537/join_particles.star -b 32 -n 20 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/10230-tau/job537/het_helix_mask_10_0_NCE/z.19.pkl
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job511/join_particles.star -b 32 -n 20 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/10230-tau/job511/het_helix_mask/z.pkl
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -b 32 -n 20 \
    --vector_path /net/jiang/scratch/li3221/Github/Unsupervised-Classification/results/10230_508_ctf_4/custom_single/pretext/feature_60.npy
python train_vector.py /net/jiang/scratch/li3221/cryodrgn2/EcACC/All_2D_refine_J384/particles.star -b 32 -n 30
python train_vector.py /net/jiang/scratch/li3221/cryodrgn2/EcACC/All_2D_refine_J384/particles.star -b 32 -n 20 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/EcACC/All_2D_refine_J384/het_helix_neighbor_2_1kld/z.29.pkl
python train_vector.py /net/jiang/scratch/li3221/Reja/dsaQtest/Class2D/job053/run_it025_data.star -b 32 -n 30
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10243-tau/Class2D/job286/run_it025_data.star -n 100
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10243-tau/Class2D/job318/run_it025_data.star -n 100



# train_TT.py
python train_TT.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation --datadir /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2 --cylinder_mask 192 --center_mask 192 -b 2 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/noise/ctf.pkl
python train_TT.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi --cylinder_mask 192 --center_mask 192 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randpsi/noise/ctf.pkl
python train_TT.py /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi10/join_particles.star -n 100 --simulation --datadir /net/jiang/scratch/li3221/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi10 --cylinder_mask 192 --center_mask 192 --ctf_path /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven_randshift_randpsi10/noise/ctf.pkl

python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 20 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job508/ctf.pkl --center_mask 64 --cylinder_mask 32
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/join_particles.star --cylinder_mask 96 -n 100 --ctf_path /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/ctf.pkl
python train_TT.py /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job505/join_particles.star -n 20 --ctf_path /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job505/ctf.pkl --center_mask 96 --cylinder_mask 64
python train_TT.py /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job509/join_particles.star -n 100 --ctf_path /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job509/ctf.pkl --center_mask 96 --cylinder_mask 64
python train_TT.py /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job505/FiT_join_particles_0.star -n 20 --ctf_path /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job505/join_particles/ctf.pkl --center_mask 96 --cylinder_mask 64
python train_TT.py /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job509/FiT_join_particles_0.star -n 100 --ctf_path /net/jiang/scratch/li3221/filament_cluster/10340/JoinStar/job509/join_particles/ctf.pkl --center_mask 96 --cylinder_mask 64

python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/join_particles.star --cylinder_mask 96 -n 50 --ctf_path /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/ctf.pkl
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 100 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job508/ctf.pkl --center_mask 64 --cylinder_mask 32
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job308/join_particles.star --cylinder_mask 64 --center_mask 96 --image_patch_size 16 -n 20 --ctf_path /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job308/ctf.pkl
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10243-tau/Class2D/job318/run_it025_data.star --cylinder_mask 32 --center_mask 96 -n 50 --ctf_path /home/jiang/li3221/scratch/practice-filament/10243-tau/Class2D/job318/ctf.pkl
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10243-tau/Class2D/job319/run_it000_data.star --cylinder_mask 32 --center_mask 96 -n 50 --ctf_path /home/jiang/li3221/scratch/practice-filament/10243-tau/Class2D/job319/ctf.pkl
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job317/join_particles.star --cylinder_mask 32 --center_mask 96 -n 50 --ctf_path /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job317/ctf.pkl
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10243-tau/Select/job322/particles_split1.star --cylinder_mask 32 --center_mask 96 -n 50 --ctf_path /home/jiang/li3221/scratch/practice-filament/10243-tau/Select/job322/ctf.pkl
python train_TT.py /net/jiang/scratch/li3221/cryodrgn2/EcACC/Subset_J389/particles.star --datadir /net/jiang/scratch/li3221/cryodrgn2/EcACC/Subset_J389 -n 20 --cylinder_mask 96 --center_mask 96 --ctf_path /net/jiang/scratch/li3221/cryodrgn2/EcACC/Subset_J389/ctf.pkl

python train_TT.py /home/jiang/li3221/scratch/practice-filament/10943-tau/Class2D/job142/run_it025_data.star -n 20 --ctf_path /home/jiang/li3221/scratch/practice-filament/10943-tau/Class2D/job142/ctf.pkl --cylinder_mask 64 --center_mask 96 --lazy -b 32
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10943-tau/Class2D/job149/run_it025_data.star -n 20 --ctf_path /home/jiang/li3221/scratch/practice-filament/10943-tau/Class2D/job149/ctf.pkl --cylinder_mask 32 --center_mask 96 --lazy -b 32

python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 100 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job508/ctf.pkl --center_mask 96 --cylinder_mask 32 --lazy

python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/Extract/job619/particles.star -n 20 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/Extract/job619/ctf.pkl --center_mask 96 --cylinder_mask 96
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/Class2D/job622/run_it025_data.star -n 100 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/Class2D/job622/ctf.pkl --center_mask 96 --cylinder_mask 96
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/Class2D/job626/run_it025_data.star -n 25 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/Class2D/job626/ctf.pkl --center_mask 64 --cylinder_mask 128 --lazy
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job632/run_it025_data.star -n 20 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job632/ctf.pkl --center_mask 64 --cylinder_mask 128 --lazy
python train_TT.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 20 --ctf_path /net/jiang/scratch/li3221/practice-filament/10230-tau/JoinStar/job508/ctf.pkl --center_mask 64 --cylinder_mask 32 --lazy


python train_TT.py /home/jiang/li3221/scratch/practice-filament/10943-tau/Class2D/job127/run_it025_data.star -n 100 --ctf_path /home/jiang/li3221/scratch/practice-filament/10943-tau/Class2D/job127/ctf.pkl --center_mask 96 --cylinder_mask 96
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10943-tau/Class2D/job127/run_it025_data.star -n 100

# train_2D_original.py
python train_2D.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/join_particles.star --cylinder_mask 128 --image_patch_size 16 -n 20 --ctf_path /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job607/ctf.pkl
python train_2D_original.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation --cylinder_mask 224 --center_mask 224


