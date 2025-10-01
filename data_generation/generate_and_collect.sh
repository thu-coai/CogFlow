API_PLATFORM=custom
MAIN_MODEL=ds-r1
QUICK_MODEL=ds-v3
MAIN_REFERENCE_MODEL=ds-r1 # will be used for generating the main reference, which is the baseline for determining whther the response is good. 
OTHER_REFERENCE_MODELS=ds-v3,gpt-4o # other references, will be ignored in the dataset by default
RUN_INSTANCES=2 # the number of social situation generated. 

python run_all.py \
	--platform $API_PLATFORM \
	--main_model $MAIN_MODEL \
	--quick_model $QUICK_MODEL \
	--reference_model $MAIN_REFERENCE_MODEL,$OTHER_REFERENCE_MODELS \
	--baseline_model $MAIN_REFERENCE_MODEL \
	--run_instances $RUN_INSTANCES \
	--max_workers 4

python collect_and_convert_to_dataset.py \
	--input_folders result/CogFlow_${MAIN_REFERENCE_MODEL}_6_added \
	--output_folder ../dataset \
	--reference_api $MAIN_REFERENCE_MODEL \
	--skip_apis $OTHER_REFERENCE_MODELS
