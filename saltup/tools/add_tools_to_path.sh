#!/bin/bash
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
WS_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

if [[ ":$PATH:" != *":$WS_DIR:"* ]]; then
    if [[ -z "$PATH" ]]; then
        export PATH="$WS_DIR"
    else
        export PATH="$PATH:$WS_DIR"
    fi
fi

echo "Added $WS_DIR to PATH"
