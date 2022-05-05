<<<<<<< HEAD
# stock-shocks
Repo for MSc. Thesis

## Setup
Run all the commands in setup.sh

If using VS code  add `"python.venvPath": "~/.cache/pypoetry/virtualenvs"` to settings.json
if using pycharm 
- select the right env. In pycharm: settings>python interpreter>select the one u want (the file to select should be in `<chosen env>/bin/python`)
- add black formatter https://akshay-jain.medium.com/pycharm-black-with-formatting-on-auto-save-4797972cf5de
- set vscode keymap in settings>keymap (sinatll the plugin if vscode option is not in the dropdown menu), also add the shortcat to "clone caret above/below"

To make pystable work on ubuntu
```python
import inspect
inspect.getfile(pystable) # get pystable path
```

Change line 23 in `pystable.utils.py` to `elif "2.33" in platform_id or "2.35" in platform_id:`
then to get the last gsl version run `sudo find / -name "libgsl.so.*"`
```bash
cd pystable_path/_extensions
rm /osx-intel/libstable.so # or /macOS/i386/libstable.so
cp ubuntu/libstable.so osx-intel/libstable.so # or /macOS/i386
sudo ln /usr/lib/x86_64-linux-gnu/libgsl.so.27 /usr/lib/x86_64-linux-gnu/libgsl.so.25 # libgsl.so.23 is required but .so.your version is the current version
```

Install aws CLI
```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```


Setup dvc 
https://anno-ai.medium.com/mlops-and-data-managing-large-ml-datasets-with-dvc-and-s3-part-1-d5b8f2fb8280
## Usage
activate the virtual env  first `cd shocks && poetry shell` 

Remember to use the poetry shell everytime you need to install a new package
=======
# shocks
Repository for MSc. thesis
>>>>>>> parent of 292df0d (re-create repo)
