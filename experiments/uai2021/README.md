# Experiments
This directory contains the code for reproducing the experimental results.

# Reproducing the figures in the paper
```
$ make decompress-records
$ jupyter notebook
# Open and run Results.ipynb.
```

# Preparation
## Preparing the data
The data files need to be prepared by going into each directory in `../suite/*` and running `$ make`.
See the `README.md` in each data directory for details.

## Preparing the database
This experiment depends on MongoDB to store the intermediate results.

1. Install this repository by following the [README.md](https://github.com/takeshi-teshima/incorporating-causal-graphical-prior-knowledge-into-predictive-modeling-via-simple-data-augmentation/blob/master/README.md).

1. Install the requirements: `pip install -r requirements.txt`.

1. Install MongoDB via tarball
  ```sh
  $ wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-3.6.3.tgz
  $ tar xaf mongodb-linux-x86_64-3.6.3.tgz
  ```
  and set `path`.
  (Sidenote: the installation via homebrew / linuxbrew did not work for me).

1. Modify the contents of `scripts/config.sh` to match your local environment (to specify where the database files will be stored).

1. Run `$ scripts/mongo.sh` to start the MongoDB process (the script requires the `tmux` package. If you don't have it, install it or start mongo with your own script).

1. Create appropriate users and tables.
  ```sh
  $ mongo
  > use icml2020
  > db.createUser({user:'me',pwd:'pass',roles:[{role:'dbOwner',db:'icml2020'}]})
  > use sacred
  > db.createUser({user:'me',pwd:'pass',roles:[{role:'dbOwner',db:'sacred'}]})
  ```

### Note
- Don't install `bson` package. It has a conflicting name with the `PyMongo` package.

# Running the experiment

- Modify the experiment condition in `config/`. At least, `config/database.yml` needs to be modified for the script to run correctly.
- Run the experiment by

```bash
$ ./run_all_sachs.sh
$ ./run_all_gss.sh
$ ./run_all_boston_housing.sh
$ ./run_all_auto_mpg.sh
$ ./run_all_red_wine.sh
$ ./run_all_white_wine.sh
```

## Checking the results
1. To check the results run by Sacred, modify the contents in `scripts/omniboard.sh` and run
  ```bash
  $ scripts/omniboard.sh
  ```

2. I also used DBeaver to check the contents in the MongoDB where the results are stored.
   To setup Omniboard, run
   ```bash
   $ brew install npm
   $ npm install -g omniboard
   ```

## Formatting the results into a LaTeX table.
- Run `$ jupyter notebook` to check the script to generate the table in the paper. (The compiled table is output in `output/`).
  Originally, a MongoDB database was used, but the records are pickled under `pickle/` here.
