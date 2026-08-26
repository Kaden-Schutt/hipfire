#!/bin/bash
exec stdbuf -oL -eL /mnt/scratch/hipfire-work/ds4-mi300x-agentmaxx/target/release/examples/daemon "$@"
