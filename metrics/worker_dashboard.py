import plotly.graph_objs as go

def loss_time(vis, state):
    vis.line(state["loss"][-1:], [sum(state["compute_time"])], win="loss_time", 
        name=f"worker_{state['worker_id']}", update="append",
        opts=dict(
                showlegend=True,
                xlabel="Time, s",
                ylabel="Loss"
            ),
    )

def loss_step(vis, state):
    vis.line(state["loss"][-1:], [state["step_num"]], win="loss_step", 
        name=f"worker_{state['worker_id']}", update="append",
        opts=dict(
                showlegend=True,
                xlabel="Step",
                ylabel="Loss"
            ),
    )

def time_step(vis, state):
    vis.line(state["compute_time"][-1:], [state["step_num"]], win="time_step", 
        name=f"worker_{state['worker_id']}", update="append",
        opts=dict(
                showlegend=True,
                xlabel="Step",
                ylabel="Time, s"
            ),
    )

def time_hist(vis, state):
    if len(state["compute_time"]) > 1:
        vis.histogram(
            state["compute_time"], 
            env=state['worker_id'],
            win="step_time",
            opts=dict(
                title=f"step_time"
            )
        )

def download_time(vis, state):
    if len(state["download_time"]) > 1:
        vis.histogram(
            state["download_time"], 
            env=state['worker_id'],
            win="download_time",
            opts=dict(
                title=f"download_time"
            )

        )

    vis.line(state["download_time"][-1:], [sum(state["compute_time"])], win="downloadtime_step", 
        name=f"worker_{state['worker_id']}", update="append",
        opts=dict(
                showlegend=True,
                xlabel="Time",
                ylabel="Download time"
            ),
    )


metrics = [loss_time, loss_step, time_step, time_hist, download_time]