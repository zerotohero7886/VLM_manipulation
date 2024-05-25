# VLM_Manipulation
Manipulation in simulation and real-world environments using GPT-4 and G-DINO for commonsense image reasoning.

## Features
- **GPT-4 Modules**: Integrated for advanced image understanding and reasoning.
- **G-DINO**: Added as a tool for object detection and manipulation.
- **Mujoco Simulation**: To be attached for simulated embodiment (coming soon).

## Installation
To set up the environment, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/zerotohero7886/VLM_manipulation.git
    cd VLM_manipulation
    ```

2. **Install Poetry** (if not already installed):
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. **Install dependencies**:
    ```sh
    poetry install
    ```

## Usage
To run the agent and test functionalities, use the following command:
```sh
python3 -m src.agent
```

## Testing
To ensure everything is correctly configured and working, run the tests:
```sh
pytest -s
```


## TODO
- [x] Add GPT-4 modules
- [x] Add G-DINO (as a tool for GPT-4)
- [ ] Attach Mujoco Simulation (for simulated embodiment)

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.