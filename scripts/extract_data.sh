#!/bin/bash

function traverse() {
    names=("FLAIR_preprocessed" "Consensus")

    for file in "$1"/*
    do
        if [[ ! -d "${file}" ]] ; then
            if [[ ! "${file}" == *".gz"* ]] ; then
                echo "Skipping ${file}"
                continue
            fi

            for i in "${names[@]}"
            do
                if [[ "${file}" == *"$i"* ]]; then
                    echo "Extracting ${file}"
                    gzip -d "${file}"
                    break
                fi
            done
        else
            echo "Entering recursion with: ${file}"
            traverse "${file}"
        fi
    done
}

path="."
ds=("Pre-processed training dataset" "Unprocessed training dataset")
for zip_file in "${ds[@]}"
do
    unzip -qq -d "${zip_file}" "${zip_file}.zip"
done

traverse "$path"