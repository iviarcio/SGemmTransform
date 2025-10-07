#!/bin/bash

docker run -it --name sgemm_session \
  -v "/Users/marcio/work":/home/marcio/work \
  -v "$HOME/.zshrc":/root/.zshrc \
  -v "$HOME/.zsh_history":/root/.zsh_history \
  linux:dev zsh
