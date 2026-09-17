import pickle as pkl  
from Data_preprocessing import data_process
import streamlit as st
from cv_xtractor.extract_entities import extract_entities_from_file,extract_skills
import json
import tempfile as tf


# load model and vector 
model = pkl.load(open("Models/Model.pkl","rb"))
vector = pkl.load(open("Models/Vector.pkl","rb"))
print(model,vector)

# encode the Catagiries
Catagiries = {'ACCOUNTANT': 0, 'ADVOCATE': 1, 'AGRICULTURE': 2, 'APPAREL': 3, 'ARTS': 4, 'AUTOMOBILE': 5, 'AVIATION': 6, 'BANKING': 7, 'BPO': 8, 'BUSINESS-DEVELOPMENT': 9, 'CHEF': 10, 'CONSTRUCTION': 11, 'CONSULTANT': 12, 'DESIGNER': 13, 'DIGITAL-MEDIA': 14, 'ENGINEERING': 15, 'FINANCE': 16, 'FITNESS': 17, 'HEALTHCARE': 18, 'HR': 19, 'INFORMATION-TECHNOLOGY': 20, 'PUBLIC-RELATIONS': 21, 'SALES': 22, 'TEACHER': 23}



def cv_data_extractor(user_cv):

    with tf.NamedTemporaryFile(delete=False,suffix=".pdf") as tf_file:
        tf_file.write(user_cv.getvalue())
        cv_path = tf_file.name
    

    try:
        text = extract_entities_from_file(cv_path)
        text = json.dumps(text,indent=4)
        return(text)
    except Exception as e:
        st.warning(f"Sorry we Face Following Error : {e}")


def model_pred(user_input):  
    feature = cv_data_extractor(user_input)
    # apply data_preprocess on user input 
    feature = data_process(feature)
    # apply vectorization 
    vector_feature = vector.transform([feature])
    print(vector_feature.shape)
    # apply model 
    model_predict = model.predict(vector_feature)
    return model_predict[0]



# custom Css 
st.markdown(
    """
<style>
section , header {
    background: #ecf0f3 !important;
    font-family: sans-serif;
}
.stMainBlockContainer{
    max-width: 936px;
}
div[data-testid="stNumberInputContainer"] , div[data-testid="stTextInputRootElement"],div[data-baseweb="select"]{
    border: 1px solid;
}
input{
    background: #ffffff61 !important;
    }

div[data-testid="stElementContainer"]{
width: 80%;
}
div[direction="column"] {
    display: flex;
    align-items: center;
    gap: 30px;
}
.stForm{
    padding: 30px 0px;
    box-shadow:
		10px 10px 10px #d1d9e6,
		-10px -10px 10px #d1d9e6;
        padding-bottom: 40px; 
}
p {
    font-size: 15px !important;
}
h1 span {
    font-size: 35px;
     color: #4a89dc;
}
span[data-testid="stHeaderActionElements"] {
    display: none;
}
div[data-testid="stHeadingWithActionElements"]{
text-align: center;
}
div[direction="column"] > :nth-last-child(1){
width: 100% ;
}
.st-key-FormSubmitter-Input_form-Anylsis {
    width: 80% !important;
}
section [data-testid="stFileUploaderDropzone"] {
    background: rgba(151, 166, 195, 0.15) !important;
}
.stFileUploaderFile {
    background: rgba(151, 166, 195, 0.15) !important;
    padding: 5px 10px;
    border-radius: 10px;
}
@media screen and (max-width: 600px){

h1 span {
    font-size: 30px;
}

}
</style>
""",
    unsafe_allow_html=True,
)


# user_input = r"Dedicated Customer Service Manager with 15+ years of experience in Hospitality and Customer Service Management.   Respected builder and leader of customer-focused teams; strives to instill a shared, enthusiastic commitment to customer service.         Highlights         Focused on customer satisfaction  Team management  Marketing savvy  Conflict resolution techniques     Training and development  Skilled multi-tasker  Client relations specialist           Accomplishments      Missouri DOT Supervisor Training Certification  Certified by IHG in Customer Loyalty and Marketing by Segment   Hilton Worldwide General Manager Training Certification  Accomplished Trainer for cross server hospitality systems such as    Hilton OnQ  ,   Micros    Opera PMS   , Fidelio    OPERA    Reservation System (ORS) ,   Holidex    Completed courses and seminars in customer service, sales strategies, inventory control, loss prevention, safety, time management, leadership and performance assessment.        Experience      HR Administrator/Marketing Associate"


# model_predict = model_pred(user_input)
# model_predict = [key for key,value in Catagiries.items() if value == model_predict]
# print(model_predict[0])






with st.form("Input_form"):
    st.title("Resume Screening System")
    user_input = st.file_uploader("Upload Your Resume Here :",type = ["pdf"])
    submit_btn = st.form_submit_button("Anylsis")


if submit_btn:
   models_pred = model_pred(user_input)
   models_pred = [field for field ,value in Catagiries.items() if value == models_pred]
   print(models_pred[0])
   st.success(f"According to the LinearSVC model, This Resume is best suited for the   ' {models_pred[0]} '  field." )