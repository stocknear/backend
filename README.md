<div align="center">



# **Open Source Stock Analysis for Data Freaks**

<h3>

[Homepage](https://stocknear.com/) | [Discord](https://discord.com/invite/hCwZMMZ2MT)

</h3>

[![GitHub Repo stars](https://img.shields.io/github/stars/stocknear/backend)](https://github.com/stocknear/backend/stargazers)

</div>



# Techstack
This is the codebase that powers [stocknear's](https://stocknear.com/) backend, which is an open-source stock analysis & community platform.

Built with:
- [FastAPI](https://fastapi.tiangolo.com/): Python Backend
- [Fastify](https://fastify.dev/): Nodejs Backend
- [Pocketbase](https://pocketbase.io/): Database
- [Redis](https://redis.io/): Caching Data

# Getting started
Follow the instructions below to run stocknear locally on your machine.

## Prerequisites & Resources

* Python 3.x (Recommended: 3.10.12 or higher)
* Pip (Python package installer)
* PocketBase (Download and install from: https://pocketbase.io/

* Download schemas, databases and configurations files:
  * stocks.db [TODO - add link] 
  * crypto.db [TODO - add link]
  * institute.db [TODO - add link]
  * json.zip folder [TODO - add link]
  * pocketbase schema [TODO - add link]

## Installation

1. **Set up virtual env:**

`python -m venv env`

`source env/bin/activate`  # On macOS/Linux

`.\env\Scripts\activate`   # On Windows

2. **Install dependencies:**

`pip install -r requirements.txt`

## Run

`python .\app\main.py`

# Contributing
Stocknear is open-source software and you're welcome to contribute to its development.

The core idea of stocknear shall always be: ***Simplicity***, ***Maintainable***, ***Readable*** & ***Fast*** in this order.

If want to contribute to the codebase please follow these guidelines:
- Reducing complexity and increasing readability is a huge plus!
- Anything you claim is a "speedup" must be benchmarked. In general, the goal is simplicity, so even if your PR makes things marginally faster, you have to consider the tradeoff with maintainablity and readablity.
- If your PR looks "complex", is a big diff, or adds lots of lines, it won't be reviewed or merged. Consider breaking it up into smaller PRs that are individually clear wins. A common pattern I see is prerequisite refactors before adding new functionality. If you can (cleanly) refactor to the point that the feature is a 3 line change, this is great, and something easy for us to review.

# Support ❤️
If you love the idea of stocknear and want to support our mission you can help us in two ways:
- Become a [Pro Member](https://stocknear.com/pricing) of stocknear to get unlimited feature access to enjoy the platform to the fullest.
- You can donate money via [Ko-fi](https://ko-fi.com/stocknear) to help us pay the servers & data providers to keep everything running! 
