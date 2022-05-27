# Bridge AI with Bidding Information

Final project for Artificial Intelligence (2022 Spring), lectured by [Jane Yung-jen Hsu](https://iagentntu.github.io/professor/Jane.html).

## Dependencies

### Python dependencies

Install the dependencies

```shell
pip install -r ./requirements.txt
```

### Double Dummy Solver

Follow the [DDS install instruction](https://github.com/dds-bridge/dds/blob/develop/INSTALL) to build the dynamic library and put it into the `./lib` folder. There is already a [compiled file](https://github.com/JackywithaWhiteDog/Bridge-AI-with-Bidding-Information/tree/main/lib) ([v2.9.0](https://github.com/dds-bridge/dds/tree/v2.9.0)) for Linux, build it on your own if you need a specific version or for other platforms.

## Run the Simulation

Run with default settings:

```shell
python main.py
```

Run `python main.py --help` or `python main.py -h` to get the help:

```text
usage: main.py [-h] [--declarer_agent {Random,DDS}] [--defender_agent {Random,DDS}] [--num_games NUM_GAMES] [--num_cards_in_hand NUM_CARDS_IN_HAND] [--max_threads MAX_THREADS]
               [--seed SEED]

Run the matches for bridge AIs

optional arguments:
  -h, --help            show this help message and exit
  --declarer_agent {Random,DDS}
                        The name of declarer agent; currently support Random, DDS. (Default: Random)
  --defender_agent {Random,DDS}
                        The name of defender agent; currently support Random, DDS. (Default: DDS)
  --num_games NUM_GAMES
                        Number of games to play. (Default: 100)
  --num_cards_in_hand NUM_CARDS_IN_HAND
                        Number of cards in players' hands; generally be 13 or 5. (Default: 13)
  --max_threads MAX_THREADS
                        Max threads used by DDS; set to 0 for automatically deciding. (Default: 1)
  --seed SEED           Random seed for reproduction. (Default: None)
```
