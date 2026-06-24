# Cluster bring-up for swe-agent-v2 (Nemotron async RL)

Reproducible bring-up of the 8×8 H200 miles cluster used for agentic RL
experiments. Everything is driven by Ansible from **node0** (the Ray head +
control node). No patch files: our fork's code runs via a bind-mount, and miles
core changes are committed in place.

## Layout
```
cluster/
├── setup/00-control-venv.sh   # create the ansible control venv (node0)
├── inventory/hosts.ini         # real inventory (gitignored); copy from .example
├── ansible/
│   ├── ansible.cfg             # inventory + forks=8; run playbooks from here
│   ├── image.yml               # pull radixark/miles:latest on all nodes
│   ├── sync.yml                # rsync this fork -> each node's local NVMe (+reinstall)
│   ├── containers.yml          # start the miles container per node (fork mounted)
│   ├── ray.yml                 # start Ray (head then workers)
│   └── teardown.yml            # stop Ray + remove containers
```

## One-time setup
```bash
# 1. Control venv (installs ansible-core at ~/venvs/ops)
bash setup/00-control-venv.sh
source ~/venvs/ops/bin/activate

# 2. Inventory: copy the template and fill in real node IPs + ssh key
cp inventory/hosts.ini.example inventory/hosts.ini
$EDITOR inventory/hosts.ini
```

## Bring-up (run from `cluster/ansible/`)
```bash
ansible-playbook image.yml        # pull the base image (~55 GB/node, first time only)
ansible-playbook sync.yml         # push the fork to all nodes
ansible-playbook containers.yml   # start miles containers (mounts fork over /root/miles)
ansible-playbook ray.yml          # start the Ray cluster
docker exec miles ray status      # expect: 8 nodes / 64 GPU
```

## Dev loop (iterating on miles code)
Edit the fork on node0, then push to all nodes — one command:
```bash
ansible-playbook sync.yml         # rsync + re-register editable install in each container
```
Edits to existing files go live immediately (bind-mount). New/renamed files need
the reinstall `sync.yml` runs automatically when a container is up. Python won't
hot-reload a running engine/trainer — restart the job to pick up changes.

## Teardown
```bash
ansible-playbook teardown.yml                      # stop Ray + remove containers (frees GPUs)
ansible-playbook teardown.yml -e remove_image=true # also drop the base image
ansible-playbook teardown.yml -e remove_fork=true  # also drop the synced fork copy
```
Teardown never touches `/cpfs01` weights/datasets — that is a manual decision.

## Notes
- **Fork code, not the image's:** the image bakes miles at an older commit; `containers.yml`
  bind-mounts `/home/user/miles-fork` (synced by `sync.yml`) over `/root/miles` and
  re-registers the editable install so the fork is what executes.
- **Networking:** containers run `--network host` for inter-node NCCL/Ray; RoCE env
  (`NCCL_IB_HCA`, `NCCL_IB_GID_INDEX=3`, `NCCL_SOCKET_IFNAME=eth0`) is set uniformly on
  every node — the head must match the workers or the first heavy collective fails.
- **Idempotency:** `containers.yml`/`ray.yml` skip nodes already up; force a clean
  recreate/restart with `-e recreate=true` / `-e restart=true`.
