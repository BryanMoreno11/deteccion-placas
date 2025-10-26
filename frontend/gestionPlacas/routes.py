from flask import Blueprint, render_template

bp_gestion_placas = Blueprint(
    "gestion_placas", 
    __name__, 
    template_folder="templates", 
    static_folder="static",
    static_url_path="/static/gestionPlacas"
)


@bp_gestion_placas.route("/")
def index():
    return render_template("gestion_placas.html")
