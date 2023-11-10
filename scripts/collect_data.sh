#!/bin/bash

dirname='results/trace_run_prequential/intermediate_graph_abm_02__model_ABM01/test_dump'
dirname_out=${dirname}_out

# Concatenate and shuffle train and validation sets
cat ${dirname}/positive_*.jl > ${dirname}/positive.jlconcat
cat ${dirname}/negative_*.jl > ${dirname}/negative.jlconcat

shuf ${dirname}/positive.jlconcat > ${dirname}/positive.jlconcat.shuf
shuf ${dirname}/negative.jlconcat > ${dirname}/negative.jlconcat.shuf

rm ${dirname}/positive.jlconcat
rm ${dirname}/negative.jlconcat

echo "Positive samples": `wc -l ${dirname}/positive.jlconcat.shuf`
echo "Negative samples": `wc -l ${dirname}/negative.jlconcat.shuf`

# Subsample negative samples
num_positive=`wc -l ${dirname}/positive.jlconcat.shuf | awk '{print $1}'`
head -n ${num_positive} ${dirname}/negative.jlconcat.shuf > ${dirname}/negative.jlconcat.shuf.subsampled
echo "Subsampled to match positive samples:" `wc -l ${dirname}/negative.jlconcat.shuf.subsampled`

# Concatenate datasets
cat ${dirname}/negative.jlconcat.shuf.subsampled ${dirname}/positive.jlconcat.shuf > ${dirname}/all.jlconcat.shuf

# Remove shuffled files
rm ${dirname}/positive.jlconcat.shuf
rm ${dirname}/negative.jlconcat.shuf
rm ${dirname}/negative.jlconcat.shuf.subsampled

# Split train and testset
num_samples=`wc -l ${dirname}/all.jlconcat.shuf | awk '{print $1}'`
num_train=`echo "0.8 * ${num_samples}" | bc | awk '{print int($1)}'`
num_test=`echo "0.2 * ${num_samples}" | bc | awk '{print int($1)}'`

# Maybe make output directory
[ -d ${dirname_out} ] || mkdir ${dirname_out}

head -n ${num_train} ${dirname}/all.jlconcat.shuf > ${dirname_out}/train.jl
tail -n ${num_test} ${dirname}/all.jlconcat.shuf > ${dirname_out}/test.jl
