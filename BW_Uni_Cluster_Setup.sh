#Training a transcoder on BW uni cluster

# Get Miniforge and make it the main Python interpreter
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O ~/miniforge.sh
bash ~/miniforge.sh -b -p ~/miniforge
rm ~/miniforge.sh

echo "PATH=$PATH:$HOME/miniforge/bin" >> .bashrc
source .bashrc

conda init
source .bashrc


# Clone transcoder_circuits and my_rome
git clone https://github.com/LeStoe11/transcoder_circuits.git
git clone https://github.com/aip-hd-research/TransformerLens.git 

# Make sure that your Environment is properly set up with conda (requirements are found in transcoder_circuits on git)
conda env create -n myenv -f transcoder_circuits/environment.yaml
conda activate myenv

pip install -e TransformerLens

# make data repository
mkdir -p /pfs/data5/home/hd/hd_hd/hd_gx182/codellama-transcoder

# set WandB API # WandB
mkdir -p mkdir -p /pfs/data5/home/hd/hd_hd/hd_gx182/codellama-transcoder/wandb
echo "export WANDB_DIR=/nfs/data/shared/wandb" >> .bashrc
source .bashrc


# When you want to use the .sh files without changes Name your Environment "myenv", 	otherwise Change in the .sh file


# after you cloned transcoder_circuits from git to the server replace the original file in the folder sae_training with the one provided #here. 

# Important: I noticed on the last minute that you have to adjust thee file path in each python train_transcoder.py file   
#	checkpoint_path="/pfs/data5/home/hd/hd_hd/hd_pe220/codellama-transcoder", 
# to your own path (replace hd_pe220)

# Upload your job.sh and your programm.py files in the folder transcoder_circuits

# General tip when uploading from windows; make the files executable (using "chmod +x programm.py"  and remove windows line conventions "sed -i 's/\r$//' programm.py" before launching the .sh file. FOR EACH FILE INDIVIDUALLY... Otherwise you might wait for a job that returns only an error message.
# For more Details check my error log at 
# https://onedrive.live.com/redir?resid=782C107B012FF6B6%21211078&page=Edit&wd=target%28KE%20for%20Code.one%7C1b6e57b4-d85a-40ef-adab-09a716017e8b%2FBwUniCluster%7C9e1d2afd-f791-4e18-94fa-57c80ba88a7d%2F%29&wdorigin=NavigationUrl

# It is helpful to generate error and Output logs.
#	Therefore add beneath the Job requirements - replace hd_pe220 with your username;
#	#SBATCH --output=/pfs/data5/home/hd/hd_hd/hd_pe220/transcoder_circuits/job_output_j.log
#	#SBATCH --error=/pfs/data5/home/hd/hd_hd/hd_pe220/transcoder_circuits/job_error_%j.log

# Then lounch file.
# Navigate to transcoder_circuits (cd transcoder_circuits)
# From here enter  sbatch job.sh 
#	(If neccessary Change the gpu node. 
#	Available nodes are gpu_4_a100, gpu_4_h100, gpu_8, gpu_4
#	The nodes gpu_4_h100 and gpu_4 worked best for me)
# check the Status of your job by entering  squeue



# The current Status of the experiment is summarized at 
# https://onedrive.live.com/redir?resid=782C107B012FF6B6%21211078&page=Edit&wd=target%28KE%20for%20Code.one%7C1b6e57b4-d85a-40ef-adab-09a716017e8b%2FTC_eva%7Ce9e7372d-9813-4629-bbb7-f4fbb8440c53%2F%29&wdorigin=NavigationUrl

#When you read this file, your configurations are already entered here. As soon your code is running, please do the following
#1) check if the Information for the run on WandB (or in the Output file) matches the one in OneNote
#2) When the code is running, create a link with the Name and copy it to the OneNote. This is easily done by copying the Name of the run in WandB - it automatically creates a link. This allows me or someone else to inspect the run easily and maintain some overview - thanks.



