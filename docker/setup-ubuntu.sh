#!/bin/bash
set -e -u

PACKAGES=""
PACKAGES+=" vim" # Used to edit files.
PACKAGES+=" bash-completion" # Used for easy make
PACKAGES+=" curl" # Used for fetching sources.
PACKAGES+=" wget" # Used for fetching sources.
PACKAGES+=" git" # Used by the neovim build.
PACKAGES+=" libncurses5-dev" # Used by mariadb for host build part.
PACKAGES+=" openjdk-8-jdk-headless" # Used for android-sdk.
PACKAGES+=" clang-format" # Used for reformat c++ header
PACKAGES+=" graphviz" # Used for dot


DEBIAN_FRONTEND=noninteractive sudo apt-get install -yq $PACKAGES
