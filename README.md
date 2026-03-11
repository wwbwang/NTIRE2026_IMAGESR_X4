# [NTIRE 2026 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://ntire-sr.github.io/2026)

[![visitors](https://visitor-badge.laobi.icu/badge?page_id=zhengchen1999.NTIRE2026_ImageSR_x4&right_color=violet)](https://github.com/zhengchen1999/NTIRE2026_ImageSR_x4)
[![GitHub Stars](https://img.shields.io/github/stars/zhengchen1999/NTIRE2026_ImageSR_x4?style=social)](https://github.com/zhengchen1999/NTIRE2026_ImageSR_x4)

## Notice

All submitted code must follow the format defined in this repository. Submissions that do not follow the required format may be rejected during the final evaluation stage.

After the challenge ends, we will release all submitted code as open-source for reproducibility. If you would like your model to remain confidential, please contact the organizers in advance.

## How to test the baseline model?

1. `git clone https://github.com/zhengchen1999/NTIRE2026_ImageSR_x4.git`

2. Select the model you would like to test:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
   ```

   - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
   - We provide a baseline (team00): DAT (default). Switch models (default is DAT) through commenting the code in [test.py](./test.py#L19).

## How to add your model to this baseline?

> [!IMPORTANT]
>
> **🚨 Submissions that do not follow the official format will be rejected.**

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1sEliBQf27EEN2bzQUO-XZaTdVG8SYWNouKSHqRYY9mE/edit?usp=sharing) and get your team ID.
2. Put your the code of your model in folder:  `./models/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
3. Put the pretrained model in folder: `./model_zoo/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
   - Note: Please provide a download link for the pretrained model, if the file size exceeds **100 MB**. Put the link in `./model_zoo/[Your_Team_ID]_[Your_Model_Name]/[Your_Team_ID]_[Your_Model_Name].txt`: e.g. [team00_dat.txt](./model_zoo/team00_dat/team00_dat.txt)
4. Add your model to the model loader `test.py` as follows:

   - Edit the `else` to `elif` in [test.py](./test.py#L24), and then you can add your own model with model id.

   - `model_func` **must** be a function, which accept **4 params**. 

     - `model_dir`: the pretrained model. Participants are expected to save their pretrained model in `./model_zoo/` with in a folder named `[Your_Team_ID]_[Your_Model_Name]` (e.g., team00_dat). 

     - `input_path`: a folder contains several images in PNG format. 

     - `output_path`: a folder contains restored images in PNG format. Please follow the section Folder Structure. 

     - `device`: computation device.
5. Send us the command to download your code, e.g,

   - `git clone [Your repository link]`
   - We will add your code and model checkpoint to the repository after the challenge.

> [!TIP]
>
> Your model code does not need to be fully refactored to fit this repository. 
> Instead, you may add a lightweight external interface (e.g., `models/team00_DAT/io.py`) that wraps your existing code, while keeping the original implementation unchanged.
>
> Refer to previous NTIRE challenge implementations for examples: 
> https://github.com/zhengchen1999/NTIRE2025_ImageSR_x4/tree/main/models



## How to eval images using IQA metrics?

### Environments

```sh
conda create -n NTIRE-SR python=3.8
conda activate NTIRE-SR
pip install -r requirements.txt
```


### Folder Structure

```
test_dir
├── HR
│   ├── 0901.png
│   ├── 0902.png
│   ├── ...
├── LQ
│   ├── 0901x4.png
│   ├── 0902x4.png
│   ├── ...
    
output_dir
├── 0901x4.png
├── 0902x4.png
├──...

```

### Command to calculate metrics

```sh
python eval.py \
--output_folder "/path/to/your/output_dir" \
--target_folder "/path/to/test_dir/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```

The `eval.py` file accepts the following 4 parameters:

- `output_folder`: Path where the restored images are saved.
- `target_folder`: Path to the HR images in the `test` dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

### Weighted score for Perception Quality Track

We use the following equation to calculate the final weight score: 

$$
\text{Score} = \left(1 - \text{LPIPS}\right) + \left(1 - \text{DISTS}\right) + \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right).
$$

The score is calculated on the averaged IQA scores. 

## NTIRE Image SR ×4 Challenge Series

Code repositories and accompanying technical report PDFs for each edition:  

- **NTIRE 2025**: [CODE](https://github.com/zhengchen1999/NTIRE2025_ImageSR_x4) | [PDF](https://arxiv.org/pdf/2504.14582)  
- **NTIRE 2024**: [CODE](https://github.com/zhengchen1999/NTIRE2024_ImageSR_x4) | [PDF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_NTIRE_2024_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2024_paper.pdf)  
- **NTIRE 2023**: [CODE](https://github.com/zhengchen1999/NTIRE2023_ImageSR_x4) | [PDF](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhang_NTIRE_2023_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2023_paper.pdf)

## License and Acknowledgement

This code repository is release under [MIT License](LICENSE). 
