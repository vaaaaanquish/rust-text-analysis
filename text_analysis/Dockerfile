FROM rust

WORKDIR /app
RUN apt update &&\
    rm -rf ~/.cache &&\
    apt clean all &&\
    apt install -y cmake &&\
    apt install -y clang

# make dataset ./data/train.csv and ./data/test.csv
RUN mkdir /app/data
RUN wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
RUN tar -zxvf ldcc-20140209.tar.gz
COPY ./batch/run_make_dataset.py /app/run_make_dataset.py
RUN python3 run_make_dataset.py

# tool
RUN cargo install lindera-cli
RUN cargo install lindera-ipadic-neologd-builder

# make neologd
RUN apt install -y mecab libmecab-dev mecab-ipadic mecab-ipadic-utf8
RUN mkdir -p /usr/lib/x86_64-linux-gnu/mecab
RUN ln -s /var/lib/mecab/dic /usr/lib/x86_64-linux-gnu/mecab/dic
RUN curl -L https://github.com/neologd/mecab-ipadic-neologd/archive/master.zip > ./mecab-ipadic-neologd-master.zip
RUN unzip -o mecab-ipadic-neologd-master.zip
RUN ./mecab-ipadic-neologd-master/bin/install-mecab-ipadic-neologd --create_user_dic -p $(pwd)/mecab-ipadic-neologd-master/tmp -y
RUN IPADIC_VERSION=$(find ./mecab-ipadic-neologd-master/build/mecab-ipadic-*-neologd-* -type d | awk -F "-" '{print $6"-"$7}') &&\
    NEOLOGD_VERSION=$(find ./mecab-ipadic-neologd-master/build/mecab-ipadic-*-neologd-* -type d | awk -F "-" '{print $NF}') &&\
    lindera-ipadic-neologd ./mecab-ipadic-neologd-master/build/mecab-ipadic-${IPADIC_VERSION}-neologd-${NEOLOGD_VERSION} lindera-ipadic-${IPADIC_VERSION}-neologd-${NEOLOGD_VERSION}

# file
COPY ./batch/run.sh /app/run.sh
COPY ./Cargo.toml /app/Cargo.toml
COPY ./src /app/src

# build
RUN cargo build

ENTRYPOINT [ "/bin/bash" ]
