docker run -it --rm -p 8090:8090 \
                -e USER="$USER" \
                -e HOME="/home/$USER" \
                -w /home/$USER \
                -v /home/$USER/:/home/$USER/ \
                -v ~/.ssh:/root/.ssh \
                -v /scratch:/data \
                --network host \
                --gpus '"device=0"' \
                --shm-size=256gb \
                pixtrack \
                bash
