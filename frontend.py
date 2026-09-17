import streamlit as st
import requests

st.set_page_config(page_title="Academic Text Simplifier", page_icon="📚", layout="wide")

st.title("📚 Academic Text Simplifier")
st.markdown("""
Welcome to the Academic Text Simplifier! 
You can upload an academic PDF/TXT or paste text directly to simplify it using **T5 Transformer Models**.
""")

UPLOAD_API_URL = "http://127.0.0.1:8000/upload_file"
SIMPLIFY_API_URL = "http://127.0.0.1:8000/simplify"

input_mode = st.radio("Choose Input Method:", ("Upload File (PDF/TXT)", "Paste Text"))

if input_mode == "Upload File (PDF/TXT)":
    uploaded_file = st.file_uploader("Upload Academic PDF or TXT", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        st.info(f"Loaded: `{uploaded_file.name}`. Ready to process!")
        
        if st.button("Process Document", type="primary"):
            with st.spinner("Extracting and simplifying... This can take a minute for large files."):
                try:
                    # Prepare file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
                    
                    # Send POST request to FastAPI backend
                    response = requests.post(UPLOAD_API_URL, files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Processing complete!")
                        
                        st.markdown("### ✨ Simplified Summary")
                        st.info(data.get("simplified_text", "No summary found."))
                                
                    else:
                        st.error(f"Error {response.status_code} from server.")
                        st.write(response.text)
                
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please make sure you are running `python -m uvicorn app:app --reload` on port 8000.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    user_text = st.text_area("Paste the academic text you want to simplify:", height=300, placeholder="Type or paste a paragraph here...")
    
    if st.button("Simplify Text", type="primary"):
        if not user_text.strip():
            st.warning("Please paste some text first!")
        else:
            with st.spinner("Simplifying text..."):
                try:
                    response = requests.post(SIMPLIFY_API_URL, json={"text": user_text})
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success("Simplification complete!")
                        st.markdown("### ✨ Simplified Version")
                        st.info(data.get("simplified_text", "No summary found."))
                    else:
                        st.error(f"Error {response.status_code} from server.")
                        st.write(response.text)
                        
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the FastAPI backend. Please make sure you are running `python -m uvicorn app:app --reload` on port 8000.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
