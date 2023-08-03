def readDataFromS3():
    """ Reads the masterDF from S3 and returns it in a pandas DF
    
    Parameters
    ----------
    None

    Returns
    -------
    pandas DF
        a dataframe containing the masterDF
    """
    mydate = datetime.now() + timedelta(weeks= -1)
    mydate = mydate.strftime("%b%Y")
    # Set up your S3 client
    # Ideally your Access Key and Secret Access Key are stored in a file already
    # So you don't have to specify these parameters explicitly.
    s3 = boto3.client('s3',
                    aws_access_key_id=settings.***,
                    aws_secret_access_key=settings.***)
    # # Get the path to the file
    s3_response_object = s3.get_object(Bucket=settings.***, Key="***_{}.parquet".format(mydate))
    # Read your file, i.e. convert it from a stream to bytes using .read()
    df = s3_response_object['Body'].read()
    # Read your file using BytesIO
    df = pd.read_parquet(io.BytesIO(df))                                            
    
    return df

def get_previous_month_unix(ts):
    """Gets the unix timestamp of the previous month

    Parameters
    ----------
    ts : int
        Unix timestamp of first day of a month

    Returns
    -------
    int
        Unix timestamp of previous month
    """
    return (pd.Timestamp(ts, unit = 'ms') + pd.Timedelta(hours = 2) 
            - pd.DateOffset(months = 1) - pd.Timedelta(hours = 2) 
            - pd.to_datetime('1970-01-01')) // pd.Timedelta('1ms')

def convert_unix_to_date(ts):
    """Converts the unix timestamp to date string

    Parameters
    ----------
    ts : int
        Unix timestamp of first day of a month

    Returns
    -------
    str
        Date of unix timestamp
    """
    ts_converted = pd.Timestamp(ts, unit = 'ms') + pd.Timedelta(hours = 2)
    ts_converted = str(ts_converted.date())
    return ts_converted

def get_previous_month_date(ts):
    """Gets the date of the previous month

    Parameters
    ----------
    ts : int
        Unix timestamp of first day of a month

    Returns
    -------
    str
        Date of the previous month of unix timestamp 
    """
    return convert_unix_to_date(get_previous_month_unix(ts)) 
