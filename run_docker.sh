docker run -it --rm -p 8095:8095 \
                -e USER="$USER" \
                -e HOME="/home/$USER" \
                -w /home/$USER \
                -v /home/$USER/:/home/$USER/ \
                -v ~/.ssh:/root/.ssh \
                -v /mnt:/mnt \
                -v /home/wayve/saurabh/pixtrack/data:/data \
                --network host \
                --gpus '"device=0"' \
                --shm-size=256gb \
                pixtrack \
                bash
