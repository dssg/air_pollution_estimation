import sqlalchemy


def add_to_frame_table_sql(df, table, creds):

    x, y, w, h = [], [], [], []
    for vals in df['obj_bounds'].values:
        x.append(vals[0])
        y.append(vals[1])
        w.append(vals[2])
        h.append(vals[3])
    df['box_x'] = x
    df['box_y'] = y
    df['box_w'] = w
    df['box_h'] = h
    df.drop('obj_bounds', axis=1, inplace=True)

    db_host = creds['postgres']['host']
    db_name = creds['postgres']['name']
    db_user = creds['postgres']['user']
    db_pass = creds['postgres']['passphrase']

    conn = sqlalchemy.create_engine('postgresql://%s:%s@%s/%s' %
                                    (db_user, db_pass, db_host, db_name),
                                    encoding='latin1',
                                    echo=True)

    dtypes = {'obj_ind': sqlalchemy.types.INTEGER(),
              'camera_id': sqlalchemy.types.String(),
              'frame_id': sqlalchemy.types.INTEGER(),
              'datetime': sqlalchemy.DateTime(),
              'obj_classification': sqlalchemy.types.String(),
              'confidence': sqlalchemy.types.Float(precision=3, asdecimal=True),
              'video_id': sqlalchemy.types.INTEGER(),
              'box_x': sqlalchemy.types.INTEGER(),
              'box_y': sqlalchemy.types.INTEGER(),
              'box_w': sqlalchemy.types.INTEGER(),
              'box_h': sqlalchemy.types.INTEGER()}

    df.to_sql(name=table, con=conn, if_exists='append', dtype=dtypes)
