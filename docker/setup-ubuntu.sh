#!/bin/bash
set -e -u

PACKAGES=""
PACKAGES+=" vim" # Used to edit files.
PACKAGES+=" curl" # Used for fetching sources.
PACKAGES+=" git" # Used by the neovim build.
PACKAGES+=" libncurses5-dev" # Used by mariadb for host build part.
PACKAGES+=" openjdk-8-jdk-headless" # Used for android-sdk.

DEBIAN_FRONTEND=noninteractive sudo apt-get install -yq $PACKAGES
