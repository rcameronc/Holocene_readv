#!/bin/bash

## adjust for your love numbers

# Ice model

for tmax in 12010
do

for tmin in 50
do

for place in fennoscandia # europe
do

# put together file name
fileName="execute_${tmax}_${tmin}_${place}_modavg"
fileName_run="run_${tmax}_${tmin}_${place}_modavg.sh"
fileName_out="out_$(tmax)_${tmin}_${place}_modavg.out"
run_name="${tmax}_${tmin}_${place})_modavg";

# go to run folder

## create this folder in the same place as this file
mkdir run_gpr
cd run_gpr

# write an execute script that passes parameters on to execute script
rm $fileName_run

# Open file descriptor (fd) 4 for read/write on a text file.
exec 4<> $fileName_run

    # Let's print some text to fd 3
    echo "cd .." >&4
    echo "python -m memory_profiler readv_it.py --tmax $tmax --tmin $tmin --place $place" >&4
    echo "exit" >&4

# Close fd 4
exec 4>&-


## create this folder in the same directory as this file
# go to execute folder
cd ..
mkdir execute_gpr
cd execute_gpr
# rm $fileName
# write a submit script that passes parameters on to execute script

    # Open file descriptor (fd) 3 for read/write on a text file.
    exec 3<> $fileName

    # Let's print some text to fd 3
    echo "#!/bin/bash" >&3
    echo "#SBATCH -o $fileName_out" >&3
    echo "#SBATCH -A jalab" >&3
    echo "#SBATCH -J $run_name" >&3
    echo "#SBATCH --gres=gpu" >&3
    echo "#SBATCH --mem-per-cpu=125gb" >&3
    echo "#SBATCH --time=0:20:00" >&3
    echo "#SBATCH --mail-type=ALL"  >&3  # specify what kind of emails you want to get
    echo "#SBATCH --mail-user=rcreel@ldeo.columbia.edu" >&3  # specify email address"
    echo " " >&3
    # echo "module load anaconda" >&3
    echo "module load cuda80/toolkit"
    echo "source activate gpflow6_0" >&3
    echo "cd ../run_gpr/" >&3
    echo "bash ${fileName_run}" >&3
    # Close fd 3
    exec 3>&-

# submit execute file

eval "sbatch $fileName"
echo "sbatch $fileName"
#cd ../code

# go back to start
cd ..

done
done
done
