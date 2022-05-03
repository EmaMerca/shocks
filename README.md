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

```bash
cd pystable_path
rm /osx-intel/libstable.so
cp ubuntu/libstable.so osx-intel/libstable.so
sudo ln /usr/lib/x86_64-linux-gnu/libgsl.so.25 /usr/lib/x86_64-linux-gnu/libgsl.so.23 # libgsl.so.23 is required but .so.25 is the current version
```


## Usage
activate the virtual env  first `cd shocks && poetry shell` 

Remember to use the poetry shell everytime you need to install a new package
=======
# shocks
Repository for MSc. thesis
>>>>>>> parent of 292df0d (re-create repo)
