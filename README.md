# *SKKU*ter🛵 - Samsung Computer Engineering Challenge 2024
- Team name: **SKKUter**
- Affiliation: Computer Systems Lab. (CSL), Sungkyunkwan University
- Members: Junyeol Yu, Osama Khan
- E-mail: junyeol.yu@skku.edu, khan980@g.skku.edu
- Challenge site: [[link]](https://cechallenge.github.io/)

# Quick Start
## Install
To install `skkuter_op` package from the source code:
```bash
git clone https://github.com/JunyeolYu/skkuter.git
cd skkuter
pip install -r requirements.txt

cd ./skkuter_op
python3 setup.py install
```
## Usage
Assuming the model repository is available in `/path/to/model`. Use the following command to run the test script.
```bash
cd skkuter
python3 test_script -m /path/to/model -b 1 -t test_dataset.json -i skkuter
```
