import streamlit as st
import torch

nllb_model = torch.load("checkpoint.pt")

nllb_model.eval()  # disable dropout
nllb_model.cuda()

def nllb(text: str="Hello World!"):
    '''
    Run the No Language Left Behind (NLLB) model on arbitrary text
    
    This function takes an input string and returns a translation
    to the given language.

    :param text str: The text to translate
    
    :return: Results of the NLLB Algorithm
    '''

    return nllb_model.translate(text)
    

if __name__ == "__main__":
    st.title("NLLB Model")
    
    with st.sidebar:
        text = st.text_area("Translate this Text", value="Hello World!")

    result = nllb(text)

    st.text(result)

