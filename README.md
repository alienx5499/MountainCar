
<div align="center">

# 🌟 **MountainCar: OpenAI Gym Reinforcement Learning** 🌟  
### *Master the MountainCar environment with Q-Learning and Visualization*

![Build Passing](https://img.shields.io/badge/build-passing-success?style=flat-square)
![Views](https://hits.dwyl.com/alienx5499/MountainCar.svg)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/alienx5499/MountainCar/blob/main/CONTRIBUTING.md)
[![License: MIT](https://custom-icon-badges.herokuapp.com/github/license/alienx5499/mountaincar?logo=law&logoColor=white)](https://github.com/alienx5499/MountainCar/blob/main/LICENSE)
![Security](https://snyk.io/test/github/dwyl/hapi-auth-jwt2/badge.svg?targetFile=package.json)
![⭐ GitHub stars](https://img.shields.io/github/stars/alienx5499/MountainCar?style=social)
![🍴 GitHub forks](https://img.shields.io/github/forks/alienx5499/MountainCar?style=social)
![Commits](https://badgen.net/github/commits/alienx5499/MountainCar)
![🐛 GitHub issues](https://img.shields.io/github/issues/alienx5499/MountainCar)
![📂 GitHub pull requests](https://img.shields.io/github/issues-pr/alienx5499/MountainCar)
![💾 GitHub code size](https://img.shields.io/github/languages/code-size/alienx5499/MountainCar)

🔗 **[Visit the Live Demo](#)** | 📑 **[Explore Documentation](#)**

</div>

---

## **🏔️ What is MountainCar?**

MountainCar is a classic reinforcement learning environment from OpenAI Gym designed to challenge agents to master the task of driving an underpowered car up a steep mountain. This repository includes:
- **MountainCar.py**: A script for rendering and evaluating a trained Q-learning agent.
- **MountainCarAlgorithm.py**: The Q-learning implementation for training the agent.
- **Visualize_q_table.py**: Tools for analyzing and visualizing the trained Q-table.

> *"Conquer the MountainCar challenge and understand the power of Q-learning!"*

---

## **📚 Table of Contents**
1. [✨ Features](#-features)
2. [🛠️ Tech Stack](#️-tech-stack)
3. [📸 Screenshots](#-screenshots)
4. [⚙️ Setup Instructions](#️-setup-instructions)
5. [🎯 Target Audience](#-target-audience)
6. [🤝 Contributing](#-contributing)
7. [📜 License](#-license)

---

## **✨ Features**  
- 🚗 **Custom Q-Learning Algorithm**: Train agents with hyperparameters like learning rate, discount factor, and epsilon decay.
- 📈 **Q-Table Visualization**: Gain insights into the training process with histograms of Q-values.
- 💻 **Modular Codebase**: Separate scripts for training, evaluation, and visualization.
- 🏔️ **Enhanced Reward System**: Custom rewards for better learning outcomes.
- 🖥️ **Rendering Script**: Visualize the trained agent in action with real-time rendering.

---

## **🛠️ Tech Stack**

### 🌐 **Python Technologies**
- **Reinforcement Learning**: OpenAI Gym
- **Visualization**: Matplotlib, NumPy
- **Code Management**: Pickle for saving Q-tables and metrics

### 🛠️ **Scripts and Files**
- **MountainCar.py**: Script to render the agent and observe performance.
- **MountainCarAlgorithm.py**: Core Q-learning training logic.
- **Visualize_q_table.py**: Analyze and visualize Q-table and training metrics.

---

## **📸 Screenshots**
Here are visualizations showcasing the training process, Q-table analysis, and the agent in action:

1. **Total Rewards Per Episode**  
   Visualizes the total rewards collected by the agent over episodes, showing trends and improvement over time.  
   ![Total Rewards Per Episode](https://github.com/user-attachments/assets/8b4cf6f8-083c-4f9e-b136-37f157e5d892)

2. **Epsilon Decay Over Episodes**  
   Highlights how the epsilon value decreases during training, balancing exploration and exploitation.  
   ![Epsilon Decay Over Episodes](https://github.com/user-attachments/assets/c4ef1a48-45af-4820-bfb3-3412d62fcbfe)

3. **Distribution of Maximum Q-Values**  
   Demonstrates the distribution of maximum Q-values across the state space, providing insights into the agent's decision-making quality.  
   ![Distribution of Maximum Q-Values](https://github.com/user-attachments/assets/d09e7b79-bf1c-4c12-a261-1c969481d8e5)

4. **MountainCar Agent in Action**  
   Watch the trained agent perform in the MountainCar environment as it attempts to reach the goal.  
   ![MountainCar Agent in Action](https://github.com/user-attachments/assets/461d68fa-96c9-47d3-ab58-6cc98dfe8afc)  

---

## **⚙️ Setup Instructions**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/alienx5499/MountainCar.git
   ```
2. **Navigate to the Project Directory**
   ```bash
   cd MountainCar
   ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Training Script**
   ```bash
   python MountainCarAlgorithm.py
   ```
5. **Visualize Training Metrics**
   ```bash
   python Visualize_q_table.py
   ```
6. **Render the Trained Agent**
   ```bash
   python MountainCar.py
   ```

---

## **🎯 Target Audience**

1. **Reinforcement Learning Enthusiasts**: Dive deep into Q-learning and OpenAI Gym.
2. **AI Researchers**: Analyze and build upon the classic MountainCar environment.
3. **Students and Educators**: Use as a learning tool for understanding reinforcement learning.
4. **Developers**: Expand the repository with new algorithms and features.

---

## **🤝 Contributing**

We ❤️ open source! Contributions are welcome to make this project even better.  
1. Fork the repository.  
2. Create your feature branch.  
   ```bash
   git checkout -b feature/new-feature
   ```
3. Commit your changes.  
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push to the branch and open a pull request.

> Refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

---

## <img src="https://fonts.gstatic.com/s/e/notoemoji/latest/1f31f/512.webp" width="35" height="30"> Awesome Contributors

<div align="center">
	<h3>Thank you for contributing to our repository</h3><br>
	<p align="center">
		<a href="https://github.com/alienx5499/MountainCar/contributors">
			<img src="https://contrib.rocks/image?repo=alienx5499/MountainCar" width="90" height="45" />
		</a>
	</p>
</div>

---

## **📜 License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 📬 **Feedback & Suggestions**
*We value your input! Share your thoughts through [GitHub Issues](https://github.com/alienx5499/MountainCar/issues).*


🔗 **[Visit the Live Demo](#-screenshots)** | 📑 **[Explore Documentation](#)** 

---


💡 *Let's conquer the MountainCar challenge together!*

</div>
