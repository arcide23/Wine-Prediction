.PHONY: all split train graphs evaluate clean

all: split train graphs

split:
	Rscript scripts/split_wine_data.R

train:
	Rscript scripts/train_models.R

graphs:
	Rscript scripts/create_graphs.R

evaluate:
	Rscript scripts/evaluate_best_model.R

clean:
	rm -f output/*.csv output/*.pdf
