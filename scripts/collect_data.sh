#!/bin/bash

dirname='results/trace_run_prequential/intermediate_graph_abm_02__model_ABM01/test_with_obs_123/'
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
shuf ${dirname}/all.jlconcat.shuf > ${dirname}/all.jlconcat.shuf.shuf

# Remove shuffled files
rm ${dirname}/positive.jlconcat.shuf
rm ${dirname}/negative.jlconcat.shuf
rm ${dirname}/negative.jlconcat.shuf.subsampled
rm ${dirname}/all.jlconcat.shuf

# Maybe make output directory
[ -d ${dirname_out} ] || mkdir ${dirname_out}

# Copy to dir output
cp ${dirname}/all.jlconcat.shuf.shuf ${dirname_out}/all.jl
rm ${dirname}/all.jlconcat.shuf.shuf

echo 'Final dataset at ' ${dirname_out}/all.jl