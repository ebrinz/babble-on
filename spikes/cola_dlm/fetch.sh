#!/bin/zsh
# Resume each weight file until it reaches its published size (the CDN closes long transfers).
B=https://huggingface.co/ByteDance-Seed/Cola-DLM/resolve/main
get() {
  local f=$1 want=$2 tries=0
  while [ "$(stat -f %z "hf_models/$f" 2>/dev/null || echo 0)" != "$want" ]; do
    tries=$((tries+1)); [ $tries -gt 200 ] && { echo "$f gave up"; return 1; }
    curl -sSL -C - --retry 5 --speed-time 120 --speed-limit 1000 -o "hf_models/$f" "$B/$f" 2>>fetch.err
    sleep 2
  done
  echo "$f complete after $tries attempts"
}
pids=()
for kv in $(cat expected_sizes.txt); do get ${kv%%=*} ${kv##*=} & pids+=$!; done
fail=0; for p in $pids; do wait $p || fail=1; done
exit $fail
