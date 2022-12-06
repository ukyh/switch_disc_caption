Here are the similar images set constructed by different methods.
"method_CIDEr_score" contains the similar images set id constructed by CIDEr similarity;
"method_retrieval_K" contains the similar images set id constructed by VSE++ image to text retrieval with K similar images. The default setting in the paper is in "method_retrieval_05".
In each directory, there are three JSON files, i.e., similar_test.json, similar_train.json, and similar_val.json.
The format of similari images set id:
{target_image_1_id: [similar_images_id_list], target_image_2_id: [similar_images_id_list], ...}
In the JSON file, target_image_id is stored in the string type, and the similar_images_id_list is stored in the int type. Run "read_example.py", and we can see the detail.

