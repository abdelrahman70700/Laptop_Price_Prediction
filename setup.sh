mkdir -p ~/.streamlit/

echo "\
[Server]\n\
port =$port\n\
enableCORS = false\n\
headless =true \n\
\n\
" > ~/.streamlit/credentials.toml

