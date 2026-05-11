#!/bin/bash
# Download BIDMC PPG and Respiration Dataset from PhysioNet.
# https://physionet.org/content/bidmc/1.0.0/
# Safe to re-run: wget -N only fetches files newer than what's on disk.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data/bidmc_csv"
mkdir -p "$DATA_DIR"

BASE_URL="https://physionet.org/files/bidmc/1.0.0/bidmc_csv"

echo "Downloading BIDMC dataset to $DATA_DIR ..."
echo "(Subjects 01-53, three files each: Signals, Numerics, Breaths)"

for i in $(seq 1 53); do
    sid=$(printf '%02d' "$i")
    for type in Signals Numerics Breaths; do
        f="bidmc_${sid}_${type}.csv"
        out="$DATA_DIR/$f"
        if [ -f "$out" ] && [ -s "$out" ]; then
            # -N: skip if remote is not newer; use -nc as belt-and-suspenders
            wget -q -N -P "$DATA_DIR" "$BASE_URL/$f" 2>/dev/null || true
        else
            echo "  Fetching $f ..."
            wget -q --show-progress -c -O "$out" "$BASE_URL/$f"
        fi
    done
done

echo ""
echo "=== Verification ==="
MISSING=0
for i in $(seq 1 53); do
    sid=$(printf '%02d' "$i")
    f="$DATA_DIR/bidmc_${sid}_Signals.csv"
    if [ ! -s "$f" ]; then
        echo "  MISSING or empty: bidmc_${sid}_Signals.csv"
        MISSING=$((MISSING + 1))
    fi
done

TOTAL=$(ls "$DATA_DIR"/*_Signals.csv 2>/dev/null | wc -l)
if [ "$MISSING" -eq 0 ]; then
    echo "  All 53 Signals files present. Total CSV files: $(ls "$DATA_DIR"/*.csv | wc -l)"
else
    echo "  WARNING: $MISSING subject(s) missing."
    exit 1
fi
