"""Reproduce conservative event grouping for the Costa Rican amphibian GBIF TSV.
Requires pandas and numpy. Input filename: registros-anfibios.csv (tab-separated).
The full fitted event-model predictions and evaluation were generated separately.
"""
import numpy as np
import pandas as pd

d = pd.read_csv("registros-anfibios.csv", sep="\t", low_memory=False)
d = d.drop_duplicates("gbifID").dropna(subset=["decimalLatitude","decimalLongitude","species"])
d = d[d.decimalLatitude.between(8,11.3) & d.decimalLongitude.between(-86,-82.4)].copy()
d["year"] = pd.to_numeric(d.year, errors="coerce")
d = d[d.year.between(1900,2024)].copy()
d["iy"] = np.floor((d.decimalLatitude-8)/.09).astype(int)
d["ix"] = np.floor((d.decimalLongitude+86)/.092).astype(int)
d["cell"] = list(zip(d.iy,d.ix))
d["date"] = pd.to_datetime(d.eventDate,errors="coerce",utc=True,format="mixed").dt.strftime("%Y-%m-%d")
d["observer"] = d.recordedBy.fillna("").astype(str).str.strip().str.lower()
d["dataset"] = d.datasetKey.fillna("").astype(str)
valid = d.date.notna() & d.observer.ne("")
d["event_id"] = np.where(valid,d.cell.astype(str)+"|"+d.date.fillna("")+"|"+d.observer+"|"+d.dataset,d.gbifID.astype(str).radd("individual:"))
for name,frame in [("ALL",d),("HUMAN_OBSERVATION",d[d.basisOfRecord=="HUMAN_OBSERVATION"])]:
    events=frame.groupby("event_id",sort=False).agg(cell=("cell","first"),year=("year","first"),species=("species",lambda x:tuple(set(x))),records=("species","size")).reset_index()
    print(name, "records",len(frame),"events",len(events),"multi-record events",int((events.records>1).sum()))
    events.to_pickle("events_"+name.lower()+".pkl")
