--
-- PostgreSQL database dump
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner:
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner:
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: fraud; Type: TABLE; Schema: public; Owner: ; Tablespace:
--

CREATE TABLE fraud (

    body_length          text,
    channels             text,
    country              text,
    currency             text,
    delivery_method      text,
    description          text,
    email_domain         text,
    event_created        text,
    event_end            text,
    event_published      text,
    event_start          text,
    fb_published         text,
    has_analytics        text,
    has_header           text,
    has_logo             text,
    listed               text,
    name                 text,
    name_length          text,
    object_id            text,
    org_desc             text,
    org_facebook         text,
    org_name             text,
    org_twitter          text,
    payee_name           text,
    payout_type          text,
    previous_payouts     text,
    sale_duration        text,
    show_map             text,
    ticket_types         text,
    user_age             text,
    user_created         text,
    user_type            text,
    venue_address        text,
    venue_country        text,
    venue_latitude       text,
    venue_longitude      text,
    venue_name           text,
    venue_state          text,
    fraud_prob           text
);
