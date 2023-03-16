# train.py
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_1pitch/join_particles.star --simulation -n 100 --cylinder_mask 256 --center_mask 32 --image_patch_size 32
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_1p_fix/join_particles.star -n 10
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_short_test/join_particles.star -n 10
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven/join_particles.star -n 20 --simulation --cylinder_mask 192 -b 2
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation --cylinder_mask 192 -b 2
python train.py /home/jiang/li3221/scratch/simmicro/10340-tau/Noise/NoNoise_uneven2/join_particles.star -n 20 --simulation
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job508/join_particles.star -n 100
python train.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job604/join_particles.star -n 20 --image_patch_size 16
python train.py /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job245/join_particles.star -n 100
python train.py /home/jiang/li3221/scratch/practice-filament/10243-tau/JoinStar/job285/join_particles.star -n 100
python train.py /net/jiang/scratch/li3221/cryodrgn2/EcACC/Subset_J389/particles.star --datadir /net/jiang/scratch/li3221/cryodrgn2/EcACC/Subset_J389 -n 20 --cylinder_mask 128 -b 2 (Apix=5.1375)
python train.py /home/jiang/li3221/scratch/filament_cluster/10340/JoinStar/job505/join_particles.star -n 100 -b 16

# train_vector.py
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/Class2D/job087/run_it025_data.star -b 16 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job257/run_ct8_it025_data.star -b 32 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job185/run_ct14_it025_data.star -b 32 -n 100
python train_vector.py /home/jiang/li3221/scratch/filament_cluster/10340/Class2D/job166/run_it025_data.star -b 16 -n 100
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job537/join_particles.star -b 32 -n 100 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/10230-tau/job537/het_128_mask/z.pkl
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job537/join_particles.star -b 32 -n 20 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/10230-tau/job537/het_helix_mask_10_0_NCE/z.19.pkl
python train_vector.py /home/jiang/li3221/scratch/practice-filament/10230-tau/JoinStar/job511/join_particles.star -b 32 -n 20 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/10230-tau/job511/het_helix_mask/z.pkl
python train_vector.py /net/jiang/scratch/li3221/cryodrgn2/EcACC/All_2D_refine_J384/particles.star -b 32 -n 30
python train_vector.py /net/jiang/scratch/li3221/cryodrgn2/EcACC/All_2D_refine_J384/particles.star -b 32 -n 20 \
    --vector_path /net/jiang/scratch/li3221/cryodrgn2/EcACC/All_2D_refine_J384/het_helix_neighbor_2_1kld/z.29.pkl
python train_vector.py /net/jiang/scratch/li3221/Reja/dsaQtest/Class2D/job053/run_it025_data.star -b 32 -n 30




