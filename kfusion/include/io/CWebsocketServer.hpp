// #pragma once
#ifndef HEADER_CWS_SERVER
#define HEADER_CWS_SERVER
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <iostream>
#include <set>
#include <websocketpp/common/thread.hpp>
typedef websocketpp::server<websocketpp::config::asio> server;
using websocketpp::connection_hdl;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

using websocketpp::lib::thread;
using websocketpp::lib::mutex;
using websocketpp::lib::lock_guard;
using websocketpp::lib::unique_lock;
using websocketpp::lib::condition_variable;
using namespace std;

/* on_open insert connection_hdl into channel
 * on_close remove connection_hdl from channel
 * on_message queue send to all channels
 */

enum action_type {
    SUBSCRIBE,
    UNSUBSCRIBE,
    MESSAGE,
    MESSAGE_HTTP,
    MESSAGE_STR,
};

struct action {
    action(action_type t, connection_hdl h) : type(t), hdl(h) {
        ;
    }
    action(action_type t, connection_hdl h, server::message_ptr m)
      : type(t), hdl(h), msg(m) {
        ;
      }
    action(action_type t, std::string m)
      : type(t), msg_str(m) {
        ;
      }
    action_type type;
    string session_type = "ws";
    websocketpp::connection_hdl hdl;
    server::message_ptr msg;
    std::string msg_str;
    int64_t time;
};

class CWSServer {
public:
    CWSServer();
    ~CWSServer();
    void set_port(uint16_t port);
    void excute();
    void on_open(connection_hdl hdl);
    void on_close(connection_hdl hdl);
    void on_message(connection_hdl hdl, server::message_ptr msg);
    void on_http(websocketpp::connection_hdl hdl);
    action archieve_message(bool &flag);
    bool send_msg(connection_hdl &send_hdl, const std::string& msg);
    bool bind_hdl_uuid(const connection_hdl &hdl, const std::string &uuid);
    void stop_server();
    void append_msg_extern(std::string msg);
private:
    typedef std::set<connection_hdl,std::owner_less<connection_hdl> > con_list;
    std::map<websocketpp::connection_hdl, std::string, std::owner_less<connection_hdl>> m_hdl_map;
    server m_server;
    con_list m_connections;
    std::queue<action> m_actions;

    mutex m_action_lock;
    mutex m_connection_lock;
    condition_variable m_action_cond;
    uint16_t m_port;
    bool m_stop;
    mutex m_send_lock;
};

#endif