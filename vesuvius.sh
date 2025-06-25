pip3 install -r requirements.txt
ln -s /usr/local/lib/python3.10/dist-packages /usr/lib/python3.10/dist-packages
vesuvius.accept_terms --yes
git config --global user.name "jeff-j-chen"
git config --global user.email "jeffc3141@gmail.com"
apt update
apt install tmux
tmux new -s ten
tensorboard --logdir ./runs --bind_all