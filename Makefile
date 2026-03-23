.PHONY: all split train graphs clean

all: split train graphs

split:
	Rscript scripts/split_wine_data.R

train:
	Rscript scripts/train_models.R

graphs:
	Rscript scripts/create_graphs.R

clean:
	rm -f output/*.csv output/*.pdf
