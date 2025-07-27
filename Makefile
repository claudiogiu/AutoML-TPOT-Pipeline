preprocess:
	python src/preprocess.py
	@echo "Preprocessing completed."


train_pipeline: preprocess
	python src/train_pipeline.py
	@echo "Pipeline training completed."


evaluate_pipeline: train_pipeline
	python src/evaluate_pipeline.py
	@echo "Pipeline evaluation completed."


run_all: preprocess train_pipeline evaluate_pipeline
	@echo "All steps completed."