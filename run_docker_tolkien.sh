docker run -it --rm -p 8095:8095 \
                -e USER="$USER" \
                -e HOME="/home/$USER" \
                -w /home/$USER \
                -v /home/$USER/:/home/$USER/ \
                -v ~/.ssh:/root/.ssh \
                -v /data:/data \
                --network host \
                --gpus '"device=2"' \
                --shm-size=256gb \
                pixtrack \
                bash
