import streamlit as st
import sund
import numpy as np

# run app by writing 'streamlit run app_age.py' in the terminal

st.markdown('# BP aging model')

#%% let the user set some settings
st.markdown('## Set simulation time')
userage = st.number_input('Age', value=30)
simlength = st.number_input('Simulation length', value=60)
drugstart = st.number_input('Take drug at age: ', value=60)

st.markdown('## Add blood pressure values at start age')
SBP0 = st.number_input('SBP: ', value=130)
DBP0 = st.number_input('DBP: ', value=85)

st.markdown('## Add blood pressure lowering drug')
st.button("Drug off", type="primary")
if st.button("Drug on"):
    st.write("Drug is ON")
    bp_treatment = 1
else:
    st.write("Drug is OFF")
    bp_treatment = 0
    drugstart = 1


#%% find correct bp group based on sbp and dbp value
v = np.array([
    [111.472772277228, 117.860744407774, 125.223689035570, 131.612577924459],
    [112.666850018335, 119.611294462780, 124.055738907224, 133.362211221122],
    [113.166483314998, 121.500733406674, 129.834066740007, 139.556288962229],
    [114.221672167217, 124.084616795013, 133.390172350568, 142.557755775577],
    [114.584250091676, 126.251833516685, 136.390722405574, 149.169416941694],
    [117.724605793913, 128.419966996700, 139.669050238357, 153.558855885588],
    [120.168683535020, 131.002933626696, 142.947378071140, 156.558489182251],
    [121.361844517785, 133.585900256692, 144.697011367803, 157.475705903924],
    [123.529977997800, 135.335533553355, 147.420700403374, 161.170700403373],
    [124.170333700037, 135.837917125046, 148.476806013935, 167.644389438944],
    [126.197744774477, 136.893105977264, 146.893105977264, 165.3671617161719]
])

IC_DBPdata = np.array([71.7975011786893, 75.8451202263084, 80.6667452459532, 83.4641678453560])
IC_SBPdata = v[0,:]
dataage = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80])


mindiff, chosenAgeIndex = min((abs(age - userage), idx) for idx, age in enumerate(dataage))
chosenAge = dataage[chosenAgeIndex]
dataSBP = v[chosenAgeIndex, :]
mindiff, chosenColumn = min((abs(sbp - SBP0), idx) for idx, sbp in enumerate(dataSBP))

IC_DBP = IC_DBPdata[chosenColumn]
IC_SBP = IC_SBPdata[chosenColumn]



#%% install model and simulate

sund.installModel('bp_age.txt')
Modelage = sund.importModel('bp_age')
agemodel = Modelage()
t_drug = 2

drug = sund.Activity(timeunit='y')
drug.AddOutput(sund.PIECEWISE_CONSTANT, 'bp_treatment', tvalues=[drugstart-1,drugstart,drugstart+t_drug], fvalues=[0,0,bp_treatment,0])


simAge = sund.Simulation(timeunit='y', models = agemodel, activities = drug)

k1_SBP = 0.4081
k2_SBP = 0
k1_DBP = 0.9262
k2_DBP = 0.0183
bSBP = 99.8670
bDBP = 55.9788
effect_drugS = 20
effect_drugD = 5
dummy=0
theta2 = [k1_SBP, k2_SBP, k1_DBP, k2_DBP,bSBP,bDBP,IC_SBP,IC_DBP,t_drug,effect_drugS,effect_drugD,SBP0,DBP0,dummy]
agemodel.statevalues = [SBP0,DBP0]
simAge.Simulate(timevector=np.linspace(userage,userage+simlength,1000),parametervalues=theta2, resetstatesderivatives=True)

#%% plot
st.markdown('## Simulate')

plot_data4={"Time": simAge.timevector, "SBP": simAge.featuredata[:,0]}
st.line_chart(plot_data4, x="Time", y="SBP")

plot_data4={"Time": simAge.timevector, "DBP": simAge.featuredata[:,1]}
st.line_chart(plot_data4, x="Time", y="DBP")

plot_data4={"Time": simAge.timevector, "drugS": simAge.featuredata[:,2]}
st.line_chart(plot_data4, x="Time", y="drugS")

plot_data4={"Time": simAge.timevector, "ddtSBP": simAge.featuredata[:,7]}
st.line_chart(plot_data4, x="Time", y="ddtSBP")

plot_data4={"Time": simAge.timevector, "ddtDBP": simAge.featuredata[:,8]}
st.line_chart(plot_data4, x="Time", y="ddtDBP")
