import sqlalchemy as sa
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk

engine = sa.create_engine('sqlite:///:memory:') ## In memory database
connection = engine.connect()

metadata = sa.MetaData()

users = sa.Table('users', metadata,
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('name', sa.String),
    sa.Column('age', sa.Integer),
    sa.Column('job', sa.String),
)

metadata.create_all(engine)


def insert_user(name, age, job):
    query = users.insert().values(name=name, age=age, job=job)
    connection.execute(query)

def remove_user(name):
    query = users.delete().where(users.c.name == name)
    connection.execute(query)

def get_users():
    query = users.select()
    result = connection.execute(query)
    return result.fetchall()




################ TKINTER

# --- Basic Tkinter UI with three buttons and a label ---

def on_button_click(btn):
    if btn == 1:
        # add user
        insert_user(entry_name.get(), entry_age.get(), entry_job.get())
        successlabel.config(text=f"User {entry_name.get()} added!")
    elif btn == 2:
        # remove user
        remove_user(entry_name.get())
        successlabel.config(text=f"User {entry_name.get()} removed!")
    elif btn == 3:
        # show datagraph
        users = get_users()
        names = [user.name for user in users]
        ages = [user.age for user in users]
        plt.bar(names, ages)
        plt.xlabel("Names")
        plt.ylabel("Ages")
        plt.title("User Ages")
        plt.show()

root = tk.Tk()
root.title("Database App")
root.geometry("400x300")

label = tk.Label(root, text="NAME      AGE      JOB", font=("Arial", 16))
label.pack(pady=10)

## ENTRIES
entry_frame = tk.Frame(root)
entry_frame.pack(pady=10)

entry_name = tk.Entry(entry_frame, width=15)
entry_name.pack(side=tk.LEFT, padx=5)

entry_age = tk.Entry(entry_frame, width=15)
entry_age.pack(side=tk.LEFT, padx=5)

entry_job = tk.Entry(entry_frame, width=15)
entry_job.pack(side=tk.LEFT, padx=5)


## BUTTONS
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

button1 = tk.Button(button_frame, text="Add user", command=lambda: on_button_click(1))
button1.pack(side=tk.LEFT, padx=5)

button2 = tk.Button(button_frame, text="Remove user", command=lambda: on_button_click(2))
button2.pack(side=tk.LEFT, padx=5)

button3 = tk.Button(button_frame, text="Datagraph", command=lambda: on_button_click(3))
button3.pack(side=tk.LEFT, padx=5)


successlabel = tk.Label(root, text="", font=("Arial", 12))
successlabel.pack(pady=10)

root.mainloop()